import argparse
import pandas as pd
import numpy as np
from pathlib import Path


from em_evidencedh_refine import SimpleEM4EvidenceH_Refine, analyze_flips, correct_agent_judgment, analyze_flips, confusion_matrix


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

    train_df, val_df = make_train_val_split(
        df,
        case_col="test_case_id",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"#cases train={train_df['test_case_id'].nunique()}, "
          f"val={val_df['test_case_id'].nunique()}")
    print(f"#rows  train={len(train_df)}, val={len(val_df)}")

    # ===== 统计训练集中 delta_label_updated 的频率 =====
    if "delta_label_updated" in train_df.columns:
        delta_freq = train_df["delta_label_updated"].value_counts(
            dropna=False).sort_index()
        print("\n训练集中 delta_label_updated 的频率:")
        print(delta_freq)
        print(f"总计: {len(train_df)} 行")
        if len(delta_freq) > 0:
            print(f"比例:")
            for val, count in delta_freq.items():
                pct = count / len(train_df) * 100
                print(f"  {val}: {count} ({pct:.2f}%)")
    else:
        print("\n警告: 训练集中没有 delta_label_updated 列")

    # ===== 训练 refine 版 EM =====
    em = SimpleEM4EvidenceH_Refine(
        max_iter=500,
        tol=1e-4,
        seed=args.seed,
        w_gui=1.0,
        w_code=1.2,
        w_noresp=0.3,
        agent_weight=0.9,
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
    val_post = em.predict_proba(val_df)
    val_pred_step = val_df[["test_case_id", "step"]].copy()
    val_pred_step["P_EnvFail"] = val_post[:, 0]
    val_pred_step["P_AgentFail"] = val_post[:, 1] + \
        val_post[:, 2]  # 合并AgentRetryFail和AgentReasoningFail
    print("val_pred_step sample:")

    # ===== 基于 posterior 对 agent 判决做纠偏建议 =====
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

    val_correct = correct_cases_with_post(
        em, val_df, case_col="test_case_id", margin=0.0, out_dir=str(out_dir), data_dir="data")

    # 保存输出
    val_pred_step.to_csv(out_dir / "val_pred_step.csv", index=False)
    # val_case_probs.to_csv(out_dir / "val_case_probs.csv", index=False)
    val_correct.to_csv(out_dir / "val_corrected_cases.csv", index=False)


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
        elif "agent_testcase_score_x" in g.columns:
            vals = g["agent_testcase_score_x"].dropna().values
            if len(vals) > 0:
                C_case = 1.0 if vals[-1] >= 0.5 else 0.0

        if C_case is not None:
            for d in range(D):
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
        P_env = float(post[0])
        P_retry = float(post[1]) if D > 1 else 0.0
        P_reasoning = float(post[2]) if D > 2 else 0.0
        P_agent = P_retry + P_reasoning  # 保持向后兼容

        rows.append(dict(
            case_id=cid,
            P_case_EnvFail=P_env,
            P_case_AgentRetryFail=P_retry,
            P_case_AgentReasoningFail=P_reasoning,
            P_case_AgentFail=P_agent
        ))

    return pd.DataFrame(rows)


def correct_cases_with_post(em, df, case_col="test_case_id", margin=0.0, out_dir=None, data_dir="data"):
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
        P_retry = r.get("P_case_AgentRetryFail", 0.0)
        P_reasoning = r.get("P_case_AgentReasoningFail", 0.0)
        P_agent = r["P_case_AgentFail"]

        if cid == "web_0_5":
            print(
                f"C_case: {C_case}, P_env: {P_env}, P_retry: {P_retry}, P_reasoning: {P_reasoning}, P_agent: {P_agent}")

        # 确定失败类型
        agent_fail_type = None
        if P_agent > P_env + margin:
            # 区分是 retry fail 还是 reasoning fail
            if P_retry > P_reasoning:
                agent_fail_type = "AgentRetryFail"
            else:
                agent_fail_type = "AgentReasoningFail"

        if np.isnan(C_case):
            # 没原判定：直接用 argmax
            corrected = 1 if P_agent >= P_env else 0
            action = "from_model"
        elif C_case == 0:
            # agent 说 FAIL：区分 EnvFail vs AgentFail
            if P_agent > P_env + margin:
                corrected = 1
                if agent_fail_type == "AgentRetryFail":
                    action = "flip_to_AgentRetryFail"
                else:
                    action = "flip_to_AgentReasoningFail"
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
            P_case_AgentRetryFail=P_retry,
            P_case_AgentReasoningFail=P_reasoning,
            P_case_AgentFail=P_agent,
            corrected_label=corrected,
            action=action,
            agent_fail_type=agent_fail_type,
        ))

    val_df = pd.DataFrame(rows).sort_values("case_id")
    acc_original = val_df["agent_original"] == val_df["human_gt"]
    print(f"Accuracy: {acc_original.mean()}")

    acc_correct = val_df["corrected_label"] == val_df["human_gt"]
    print(f"Accuracy: {acc_correct.mean()}")
    analyze_flips(val_df, out_dir=out_dir)
    confusion_matrix(val_df)

    return pd.DataFrame(rows)


def run_prediction(df_path: str, out_dir: str, params_path: str = None):
    """
    使用已有参数进行预测

    Args:
        df_path: 数据文件路径
        out_dir: 输出目录
        params_path: 参数文件路径（JSON格式），如果为None则从out_dir中查找
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

    # 创建 EM 模型（使用默认配置）
    em = SimpleEM4EvidenceH_Refine(
        max_iter=200,
        tol=1e-4,
        seed=42,
        w_gui=1.0,
        w_code=1.2,
        w_noresp=0.5,
        # w_gui=0.0,
        # w_code=0.0,
        # w_noresp=1.0,
        agent_weight=0.9,
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
    print(f"Loaded params from {params_path}")

    post = em.predict_proba(df)

    # 构建预测结果
    pred_step = df.copy()
    pred_step["P_EnvFail"] = post[:, 0]
    pred_step["P_AgentFail"] = post[:, 1]

    pred_correct = correct_cases_with_post(
        em, df, case_col="test_case_id", margin=0.0, out_dir=out_dir, data_dir="data")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str,
    #                     default="train_em_df_labeled.xlsx")
    parser.add_argument("--data_path", type=str,
                        default="./test_em_df_labeled.xlsx")
    parser.add_argument("--test_path", type=str,
                        default="webdevjudge/traj/20251111_143620/20251111_143620_test_em_df_labeled.xlsx")
    parser.add_argument("--params_path", type=str,
                        default="em_outputs_refine_webdevjudge/em_params.json")
    parser.add_argument("--out_dir", type=str,
                        default="em_outputs_refine_webdevjudge")

    parser.add_argument("--val_ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--tau_agentfail", type=float, default=0.75)
    parser.add_argument("--tau_envfail", type=float, default=0.75)
    args = parser.parse_args()
    main(args)
    # run_prediction(args.test_path, args.out_dir, args.params_path)
