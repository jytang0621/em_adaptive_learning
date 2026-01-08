import argparse
import pandas as pd
import numpy as np
from pathlib import Path


from em_evidencedh_refine import SimpleEM4EvidenceH_Refine, analyze_flips, correct_agent_judgment, confusion_matrix, correct_agent_rootcause_terminal, correct_by_em_selective, fuse_correct_rootcause
# from em_evidence_refine_3state import SimpleEM4EvidenceH_Refine
from gate import GateModelNB, fit_gate_nb_from_step_df, load_gate_model, build_case_features, binarize_features, infer_gate_nb_from_step_df, gate_predict_proba_nb, save_gate_model
from evaluation_utils import calculate_and_print_accuracy

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix in [".csv"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file: {path}")
    return df


def make_train_val_split(df: pd.DataFrame,
                         case_col: str = "test_case_id",
                         val_ratio: float = 0.2,
                         seed: int = 42):
    """按 case 粒度划分 train/val"""
    rng = np.random.default_rng(seed)
    case_ids = df[case_col].astype(str).unique()
    rng.shuffle(case_ids)
    n_val = int(len(case_ids) * val_ratio)
    val_ids = set(case_ids[:n_val])
    train_ids = set(case_ids[n_val:])
    train_df = df[df[case_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[case_col].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def aggregate_case_posterior(em, df_case):
    eps = 1e-9
    # 1) 基于 step-level 证据的 log-likelihood
    log_like = np.zeros(2)  # [EnvFail, AgentFail]
    for _, r in df_case.iterrows():
        for d in (0, 1):
            p_gui = em.theta[d, 0]
            p_code = em.theta[d, 1]
            p_no = em.theta[d, 2]

            # 只对未 mask 的通道计入；你数据里的 M_* 已经控制好哪些步看哪些证据
            if r["M_gui"] == 0:
                e = r["E1_gui"]
                log_like[d] += np.log((p_gui if e == 1 else 1-p_gui) + eps)
            if r["M_code"] == 0:
                e = r["E2_code"]
                log_like[d] += np.log((p_code if e == 1 else 1-p_code) + eps)
            if r["M_noresp"] == 0:
                e = r["E4_noresp"]
                log_like[d] += np.log((p_no if e == 1 else 1-p_no) + eps)

    # 2) C_case 通道 (用 EM 学出来的 psi)
    C_vals = df_case["agent_testcase_score_x"].dropna().values
    if len(C_vals):
        C_case = 1.0 if C_vals[-1] >= 0.5 else 0.0
        for d in (0, 1):
            psi_d = em.psi[d]
            log_like[d] += np.log(
                (psi_d if C_case == 1 else 1-psi_d) + eps
            )

    # 3) 加上 prior P_delta，归一化成 posterior
    log_post = np.log(em.p_delta + eps) + log_like
    m = log_post.max()
    post = np.exp(log_post - m)
    post = post / post.sum()
    P_env, P_agent = post[0], post[1]
    return P_env, P_agent


def aggregate_case_probs(df: pd.DataFrame,
                         post: np.ndarray,
                         case_col: str = "test_case_id",
                         alpha: float = 0.75,
                         reflect_weight: float = 2.0,
                         em: SimpleEM4EvidenceH_Refine = None) -> pd.DataFrame:
    """把 step-level posterior 聚合到 case-level"""
    tmp = df.copy()
    p_env, p_agent = aggregate_case_posterior(em, tmp)

    tmp["P_EnvFail"] = p_env
    tmp["P_AgentFail"] = p_agent

    rows = []
    for cid, g in tmp.groupby(case_col):
        q = np.clip(g["P_AgentFail"].values, 0.0, 1.0)
        P_case_AgentFail = 1.0 - float(np.prod((1.0 - q) ** alpha))
        P_case_EnvFail = 1.0 - P_case_AgentFail
        rows.append(dict(
            case_id=cid,
            P_case_EnvFail=P_case_EnvFail,
            P_case_AgentFail=P_case_AgentFail,
        ))
    return pd.DataFrame(rows)

    #


def main(args):
    df = load_data(args.data_path)
    for col in ["E1_gui", "E2_code"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if "E4_noresp" not in df.columns:
        df["E4_noresp"] = 0.0
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "agent_testcase_score_x" not in df.columns:
        df["agent_testcase_score_x"] = df["agent_testcase_score"]

    if "delta_label" not in df.columns:
        df["delta_label"] = np.nan
    if "delta_label_updated" in df.columns:
        df["delta_label"] = df["delta_label_updated"]

    # train_df, val_df = make_train_val_split(
    #     df,
    #     case_col="test_case_id",
    #     val_ratio=args.val_ratio,
    #     seed=args.seed,
    # )

    # print(f"#cases train={train_df['test_case_id'].nunique()}, "
    #       f"val={val_df['test_case_id'].nunique()}")
    # print(f"#rows  train={len(train_df)}, val={len(val_df)}")

    # 使用全部数据进行训练
    train_df = df

    # ===== 训练 refine 版 EM =====
    gate, report_train = fit_gate_nb_from_step_df(train_df)
    gate_model_path = Path(args.out_dir) / "gate_model.pkl"
    save_gate_model(gate, gate_model_path)

    em = SimpleEM4EvidenceH_Refine(
        max_iter=200,
        tol=1e-4,
        seed=42,
        w_gui=args.w_gui,
        w_code=args.w_code,
        w_noresp=args.w_noresp,
        agent_weight=args.agent_weight,
        a_pi=5.0,
        b_pi=5.0,
        a_c0=3.0,
        b_c0=3.0,
        a_c1=3.0,
        b_c1=3.0,
        theta_floor=0.05,
        theta_ceil=0.95,
        pi_floor=0.02,
        temp=0.8,
    )
    em.fit(
        train_df,
        col_case="test_case_id",
        col_agent="agent_testcase_score_x",
        col_delta="delta_label",
        col_gui="E1_gui",
        col_code="E2_code",
        col_noresp="E4_noresp",
        col_w="weight",
    )

    params = em.get_params()
    print("params", params)

    # 保存参数
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "em_params.json", 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"Params saved to {out_dir / 'em_params.json'}")

    # ===== 验证集 step-level 预测 =====
    # val_post = em.predict_proba(val_df)
    # val_pred_step = val_df[["test_case_id", "step"]].copy()
    # val_pred_step["P_EnvFail"] = val_post[:, 0]
    # val_pred_step["P_AgentFail"] = val_post[:, 1]
    # print("val_pred_step sample:")

    # # ===== 基于 posterior 对 agent 判决做纠偏建议 =====
    # val_correct = correct_agent_judgment(
    #     val_df,
    #     em,
    #     tau_agentfail=args.tau_agentfail,
    #     tau_envfail=args.tau_envfail,
    #     tau_envfail_high=args.tau_envfail,
    #     alpha=0.75,
    #     col_case="test_case_id",
    #     col_agent="agent_testcase_score_x",
    # )

    # # val_correct = correct_cases_with_post(
    # #     em, val_df, case_col="test_case_id", margin=0.0, out_dir=str(out_dir))

    # # 保存输出
    # val_pred_step.to_csv(out_dir / "val_pred_step.csv", index=False)
    # # val_case_probs.to_csv(out_dir / "val_case_probs.csv", index=False)
    # val_correct.to_csv(out_dir / "val_corrected_cases.csv", index=False)


def aggregate_case_posteriors(em, df, case_col="test_case_id"):
    """
    用 em 的生成参数，对每个 case 做一次完整贝叶斯：
    P(delta | 所有 step 证据 + agent_testcase_score)

    返回 DataFrame:
      [case_id, P_case_EnvFail, P_case_AgentFail]
    """
    eps = 1e-9
    rows = []
    D = em.p_delta.shape[0]

    for cid, g in df.groupby(case_col):
        if cid == "web_0_5":
            print(f"cid: {cid}")
        g = g.sort_index()

        # 1) step-level 证据 log-likelihood
        # log_like = np.zeros(2)  # [EnvFail, AgentFail]
        log_like = np.zeros(D)

        for _, r in g.iterrows():
            # for d in (0, 1):
            for d in range(D):
                p_gui = em.theta[d, 0]
                p_code = em.theta[d, 1]
                p_no = em.theta[d, 2]

                # gui 通道
                if ("M_gui" not in g.columns) or (r["M_gui"] == 0):
                    e = float(r["E1_gui"])
                    log_like[d] += np.log((p_gui if e ==
                                          1.0 else 1 - p_gui) + eps)

                # code 通道
                if ("M_code" not in g.columns) or (r["M_code"] == 0):
                    e = float(r["E2_code"])
                    log_like[d] += np.log((p_code if e ==
                                          1.0 else 1 - p_code) + eps)

                # noresp 通道
                if ("M_noresp" not in g.columns) or (r["M_noresp"] == 0):
                    e = float(r["E4_noresp"])
                    log_like[d] += np.log((p_no if e ==
                                          1.0 else 1 - p_no) + eps)

        # 2) agent_testcase_score 作为 C_case 通道
        C_case = None
        if "agent_testcase_score" in g.columns:
            vals = g["agent_testcase_score"].dropna().values
            if len(vals) > 0:
                C_case = 1.0 if vals[-1] >= 0.5 else 0.0

        if C_case is not None:
            for d in (0, 1):
                psi_d = float(em.psi[d])
                if C_case == 1.0:
                    log_like[d] += np.log(psi_d + eps)
                else:
                    log_like[d] += np.log(1 - psi_d + eps)

        # 3) prior + 归一化 → posterior
        log_post = np.log(em.p_delta + eps) + log_like
        m = log_post.max()
        post = np.exp(log_post - m)
        post = post / post.sum()
        P_env, P_agent = float(post[0]), float(post[1])

        rows.append(dict(
            case_id=cid,
            P_case_EnvFail=P_env,
            P_case_AgentFail=P_agent
        ))

    return pd.DataFrame(rows)


def correct_cases_with_post(em, df, case_col="test_case_id", margin=0.0, out_dir=None):
    case_probs = aggregate_case_posteriors(em, df, case_col)

    # 拿到每个 case 的 agent 原始判定（最后一条）
    agent_orig = (df
                  .sort_index()
                  .groupby(case_col)["agent_testcase_score_x"]
                  .last()
                  .rename("agent_original"))

    out = (case_probs
           .set_index("case_id")
           .join(agent_orig)
           .reset_index())

    rows = []
    for _, r in out.iterrows():
        cid = r["case_id"]
        C_case = r["agent_original"]  # 0/1 or NaN
        P_env = r["P_case_EnvFail"]
        P_agent = r["P_case_AgentFail"]

        if cid == "web_0_5":
            print(f"C_case: {C_case}, P_env: {P_env}, P_agent: {P_agent}")
        if np.isnan(C_case):
            # 没原判定：直接用 argmax
            corrected = 1 if P_agent >= P_env else 0
            action = "from_model"
        elif C_case == 0:
            # agent 说 FAIL：区分 EnvFail vs AgentFail
            if P_agent > P_env + margin:
                corrected = 1
                action = "flip_to_AgentFail"
            else:
                corrected = 0
                action = "keep_EnvFail_or_AgentFail"
        else:  # C_case == 1, agent 说 PASS
            # 只在强 EnvFail 证据下翻转
            if P_env > P_agent + margin:
                corrected = 0
                action = "flip_to_EnvFail"
            else:
                corrected = 1
                action = "keep_AgentJudge"

        rows.append(dict(
            case_id=cid,
            human_gt=df[df["test_case_id"] == cid]["phi"].dropna().values[-1],
            agent_original=C_case,
            P_case_EnvFail=P_env,
            P_case_AgentFail=P_agent,
            corrected_label=corrected,
            action=action,
        ))

    val_df = pd.DataFrame(rows).sort_values("case_id")
    acc_original = val_df["agent_original"] == val_df["human_gt"]
    print(f"Accuracy: {acc_original.mean()}")

    acc_correct = val_df["corrected_label"] == val_df["human_gt"]
    print(f"Accuracy: {acc_correct.mean()}")
    analyze_flips(val_df, out_dir=out_dir)
    confusion_matrix(val_df)

    return pd.DataFrame(rows)


def build_case_df_with_risk(df: pd.DataFrame, gate_model_path: Path) -> pd.DataFrame:
    """
    使用gate模型构建case-level数据框并计算风险等级
    
    Args:
        df: step-level数据框
        gate_model_path: gate模型文件路径
    
    Returns:
        case_df_test: 包含test_case_id, p_err, gt, agent_pred, risk的case-level数据框
    """
    print(f"Loading gate model from: {gate_model_path}")
    gate_nb = load_gate_model(gate_model_path)
    
    # 构建case-level特征
    X_case_test = build_case_features(
        df,
        col_case="test_case_id",
        col_step="step",
        evidence_cols=["E1_gui", "E2_code", "E4_noresp"],
        col_agent_pred="agent_testcase_score_x",
        col_gt="phi" if "phi" in df.columns else None,
    )[0]  # 只取X_case，不需要y_err和gt
    
    X_case_test_bin = binarize_features(X_case_test)
    
    # 确保特征顺序与训练时一致
    if hasattr(gate_nb, 'feature_names'):
        missing_features = set(gate_nb.feature_names) - set(X_case_test_bin.columns)
        if missing_features:
            # 添加缺失的特征（填充0）
            for feat in missing_features:
                X_case_test_bin[feat] = 0
        X_case_test_bin = X_case_test_bin[gate_nb.feature_names]
    
    # 预测p_err
    p_err = gate_predict_proba_nb(gate_nb, X_case_test_bin).values
    
    # 创建case-level数据框
    case_df_test = pd.DataFrame({
        "test_case_id": X_case_test.index,
        "p_err": p_err,
        "gt": X_case_test["phi"],
        "agent_pred": X_case_test["agent_pred"] if "agent_pred" in X_case_test.columns else 
                     df.groupby("test_case_id")["agent_testcase_score_x"].first().reindex(X_case_test.index).fillna(0).astype(int)
    })
    
    # ===== Step 2: 分层 =====
    case_df_test["risk"] = pd.cut(
        case_df_test["p_err"],
        bins=[0.0, 0.4, 0.6, 1.0],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    
    return case_df_test


def apply_risk_based_correction(case_df_test: pd.DataFrame, em, df: pd.DataFrame) -> pd.DataFrame:
    """
    根据风险等级对case进行矫正
    
    Args:
        case_df_test: case-level数据框，包含test_case_id, p_err, gt, agent_pred, risk
        em: EM模型
        df: step-level数据框
    
    Returns:
        pred_correct: 包含矫正结果的DataFrame，包含case_id, agent_original, corrected_label, 
                     action, risk, p_err, P_case_AgentFail, P_case_EnvFail等
    """
    # ===== Step 3: 只对 mid/high 做 EM =====
    rows = []
    for case in case_df_test.itertuples():
        if case.risk == "low":
            final = case.agent_pred
            decision = "KEEP"
            rows.append({
                "case_id": case.test_case_id,
                "agent_original": case.agent_pred,
                "corrected_label": final,
                "action": decision,
                "risk": case.risk,
                "p_err": case.p_err,
                "gt": case.gt,
            })
            continue
        
        p_agent = em_predict(case, em, df)
        p_env = 1 - p_agent
        
        if case.risk == "mid":
            if p_agent > 0.7:
                final = 1 - case.agent_pred
                decision = "FLIP"
            else:
                final = case.agent_pred
                decision = "KEEP"
        
        elif case.risk == "high":
            if p_agent > 0.7:
                final = 1-case.agent_pred
                decision = "FLIP"
            else:
                final = 1 - case.agent_pred
                decision = "FLIP"
               
        
        rows.append({
            "case_id": case.test_case_id,
            "agent_original": case.agent_pred,
            "corrected_label": final,
            "action": decision,
            "risk": case.risk,
            "p_err": case.p_err,
            "P_case_AgentFail": p_agent,
            "P_case_EnvFail": p_env,
        })
    
    pred_correct = pd.DataFrame(rows)
    return pred_correct


def run_prediction(df_path: str, out_dir: str, params_path: str = None, args=None):
    """
    使用已有参数进行预测

    Args:
        df_path: 数据文件路径
        out_dir: 输出目录
        params_path: 参数文件路径（JSON格式），如果为None则从out_dir中查找
        args: 命令行参数，包含 w_gui, w_code, w_noresp, agent_weight
    """
    import json

    # 加载数据
    df = load_data(df_path)
    # df["M_code"] = 1.0
    print(df.shape)
    # 确保必要的列存在
    for col in ["E1_gui", "E2_code"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if "E4_noresp" not in df.columns:
        df["E4_noresp"] = 0.0
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "agent_testcase_score_x" not in df.columns:
        # df["agent_testcase_score_x"] = np.nan
        df["agent_testcase_score_x"] = df["agent_testcase_score"]
    if "test_case_id" not in df.columns:
        raise ValueError("Missing column: test_case_id")
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))

    # 加载参数
    if params_path is None:
        params_path = Path(out_dir) / "em_params.json"
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    print(f"Loaded params from {params_path}")

    # 从 args 获取权重参数，如果 args 为 None 则使用默认值
    w_gui = args.w_gui if args is not None else 1.0
    w_code = args.w_code if args is not None else 1.2
    w_noresp = args.w_noresp if args is not None else 0.3
    agent_weight = args.agent_weight if args is not None else 0.9

    # 创建 EM 模型（使用默认配置）
    
    em = SimpleEM4EvidenceH_Refine(
        max_iter=200,
        tol=1e-4,
        seed=42,
        w_gui=w_gui,
        w_code=w_code,
        w_noresp=w_noresp,
        agent_weight=agent_weight,
        a_pi=5.0,
        b_pi=5.0,
        a_c0=3.0,
        b_c0=3.0,
        a_c1=3.0,
        b_c1=3.0,
        theta_floor=0.05,
        theta_ceil=0.95,
        pi_floor=0.02,
        temp=0.8,
    )

    # 加载已有参数
    em.load_params(params)
   
    post = em.predict_proba(df)
    df_tmp = df.copy()
    # df_tmp["P_pass"] = post[:,2]
    df_tmp["P_fail_env"] = post[:,0]
    df_tmp["P_fail_agent"] = post[:,1]

    E, w, M, case_ids, C_raw, _ = em._extract(df)

    df_tmp["E_code"] = E[:,1]
    
    # 构建预测结果
    pred_step = df.copy()
    pred_step["P_EnvFail"] = post[:, 0]
    pred_step["P_AgentFail"] = post[:, 1]

    
    # 从 args 获取 tau 参数，如果 args 为 None 则使用默认值
    tau_agentfail = args.tau_agentfail if args is not None else 0.75
    tau_envfail = args.tau_envfail if args is not None else 0.75
    
    pred_correct = correct_agent_judgment(
            df,
            em,
            tau_agentfail=tau_agentfail,
            tau_envfail=tau_envfail,
            tau_envfail_high=tau_envfail,
            alpha=0.75,
            col_case="test_case_id",
            col_agent="agent_testcase_score_x",
        )

    # ===== Step 1: 批量算 p_err =====
    # 尝试加载gate模型
    gate_model_path = None
    if args is not None and hasattr(args, 'gate_model_path') and args.gate_model_path:
        gate_model_path = Path(args.gate_model_path)
    else:
        # 默认从out_dir查找gate模型
        gate_model_path = Path(out_dir) / "gate_model.pkl"
        if not gate_model_path.exists():
            # 尝试从上级目录查找
            gate_model_path = Path(out_dir).parent / "gate_model.pkl"
    
    if gate_model_path.exists():
        # 使用独立函数构建case-level数据框并计算风险等级
        case_df_test = build_case_df_with_risk(df, gate_model_path)
        
        # 使用独立函数根据风险等级进行矫正
        pred_correct = apply_risk_based_correction(case_df_test, em, df)
        
        # 计算并打印准确率
        calculate_and_print_accuracy(pred_correct, df, gt_col="phi", case_col="test_case_id", prefix="Gate-based correction")
    else:
        print(f"Gate model not found at {gate_model_path}, using default correct_agent_judgment")
        
        
    # 保存结果
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_step.to_csv(out_dir / "pred_step.csv", index=False)
    if pred_correct is not None:
        pred_correct.to_csv(out_dir / "pred_corrected_cases.csv", index=False)

    print(f"Predictions saved to {out_dir}")
    print(f"Step-level predictions: {len(pred_step)} rows")
    if pred_correct is not None:
        print(f"Corrected cases: {len(pred_correct)} cases")

    return pred_step, pred_correct


def em_predict(case_row, em, df):
    """
    对单个case使用EM模型预测AgentFail概率
    
    Args:
        case_row: case的行数据（包含case_id等信息）
        em: EM模型
        df: step-level数据框
    
    Returns:
        p_agent: P(AgentFail)概率
    """
    case_id = case_row.test_case_id if hasattr(case_row, 'test_case_id') else case_row.case_id
    case_df = df[df["test_case_id"] == case_id]
    if len(case_df) == 0:
        return 0.5  # 默认值
    
    post = em.predict_proba(case_df)
    # 取最后一步的AgentFail概率
    p_agent = float(post[-1, 1]) if post.shape[1] >= 2 else 0.5
    return p_agent

def has_direction_support(case_row, df):
    """
    判断case是否有方向性支持（可以根据需要实现具体逻辑）
    这里先返回一个简单的实现
    
    Args:
        case_row: case的行数据
        df: step-level数据框
    
    Returns:
        bool: 是否有方向性支持
    """
    # TODO: 实现具体逻辑
    # 可以根据实际需求实现，比如检查证据的一致性等
    # 这里先返回True作为占位符
    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        # default="/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_claude_4_train.xlsx")
                        default="/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_webdevjudge_ui_tars_train.xlsx")
    parser.add_argument("--test_path", type=str,
                        # default="/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_claude_4_test.xlsx")
                        default="/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_webdevjudge_ui_tars_test.xlsx")
    parser.add_argument("--params_path", type=str,
                        default="/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_claude_4/em_params.json")
    parser.add_argument("--out_dir", type=str,
                        default="/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_claude_4")

    parser.add_argument("--val_ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--tau_agentfail", type=float, default=1)
    parser.add_argument("--tau_envfail", type=float, default=0.65)

    # EM weight parameters
    parser.add_argument("--w_gui", type=float, default=0.8,
                        help="Weight for GUI evidence channel")
    parser.add_argument("--w_code", type=float, default=1.2,
                        help="Weight for code evidence channel")
    parser.add_argument("--w_noresp", type=float, default=0.3,
                        help="Weight for no-response evidence channel")
    parser.add_argument("--agent_weight", type=float, default=0.5,
                        help="Weight for agent judgment")

    # parser.add_argument("--val_ratio", type=float, default=0.4)
    # parser.add_argument("--seed", type=int, default=127)
    # parser.add_argument("--tau_agentfail", type=float, default=0.75)
    # parser.add_argument("--tau_envfail", type=float, default=0.75)

    # # EM weight parameters
    # parser.add_argument("--w_gui", type=float, default=1,
    #                     help="Weight for GUI evidence channel")
    # parser.add_argument("--w_code", type=float, default=1.3,
    #                     help="Weight for code evidence channel")
    # parser.add_argument("--w_noresp", type=float, default=1.3,
    #                     help="Weight for no-response evidence channel")
    # parser.add_argument("--agent_weight", type=float, default=0.9,
    #                     help="Weight for agent judgment")

    args = parser.parse_args()
    main(args)
    run_prediction(args.test_path, args.out_dir, args.params_path, args)
