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

    # 获取 EM 的通道权重（与训练阶段保持一致）
    w_gui = getattr(em, 'w_gui', 1.0)
    w_code = getattr(em, 'w_code', 1.0)
    w_no = getattr(em, 'w_no', 0.5)
    agent_weight = getattr(em, 'agent_weight', 1.0)

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
                log_p_gui = np.log((p_gui if e == 1 else 1-p_gui) + eps)
                log_like[d] += w_gui * log_p_gui
            if r["M_code"] == 0:
                e = r["E2_code"]
                log_p_code = np.log((p_code if e == 1 else 1-p_code) + eps)
                log_like[d] += w_code * log_p_code
            if r["M_noresp"] == 0:
                e = r["E4_noresp"]
                log_p_no = np.log((p_no if e == 1 else 1-p_no) + eps)
                log_like[d] += w_no * log_p_no

    # 2) C_case 通道 (用 EM 学出来的 psi)
    C_vals = df_case["agent_testcase_score_x"].dropna().values
    if len(C_vals):
        C_case = 1.0 if C_vals[-1] >= 0.5 else 0.0
        for d in (0, 1):
            psi_d = em.psi[d]
            log_p_agent = np.log((psi_d if C_case == 1 else 1-psi_d) + eps)
            log_like[d] += agent_weight * log_p_agent

    # 3) 加上 prior P_delta，归一化成 posterior
    log_post = np.log(em.p_delta + eps) + log_like
    m = log_post.max()
    post = np.exp(log_post - m)
    post = post / post.sum()
    P_env, P_agent = post[0], post[1]
    return P_env, P_agent


def aggregate_case_probs(df: pd.DataFrame,
                         post: np.ndarray = None,
                         case_col: str = "test_case_id",
                         alpha: float = 0.75,
                         reflect_weight: float = 2.0,
                         em: SimpleEM4EvidenceH_Refine = None) -> pd.DataFrame:
    """把 step-level posterior 聚合到 case-level

    修复：按 case 循环调用 aggregate_case_posterior，而不是对整个 df 算一次
    """
    rows = []
    for cid, g in df.groupby(case_col):
        # 为每个 case 单独计算 posterior
        P_case_EnvFail, P_case_AgentFail = aggregate_case_posterior(em, g)
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

    # ===== 训练 refine 版 EM =====
    em = SimpleEM4EvidenceH_Refine(
        max_iter=500,
        tol=1e-4,
        seed=args.seed,
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
        temp=args.temp,
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
    val_pred_step["P_AgentFail"] = val_post[:, 1]
    print("val_pred_step sample:")

    # ===== 基于 posterior 对 agent 判决做纠偏建议 =====
    val_correct = correct_agent_judgment(
        val_df,
        em,
        margin_agentfail=args.margin_agentfail,
        tau_envfail_high=args.tau_envfail_high,
        tau_agentfail_high=args.tau_agentfail_high,
        tau_agentfail_floor=args.tau_agentfail_floor,
        tau_support=args.tau_support,
        tau_support_pos=args.tau_support_pos,
        tau_support_neg=args.tau_support_neg,
        min_neg_channels=args.min_neg_channels,
        col_case="test_case_id",
        col_agent="agent_testcase_score_x",
        out_dir=str(out_dir),
    )

    # val_correct = correct_cases_with_post(
    #     em, val_df, case_col="test_case_id", margin=0.0, out_dir=str(out_dir))

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

    # 获取 EM 的通道权重（与训练阶段保持一致）
    w_gui = getattr(em, 'w_gui', 1.0)
    w_code = getattr(em, 'w_code', 1.0)
    w_no = getattr(em, 'w_no', 0.5)
    agent_weight = getattr(em, 'agent_weight', 1.0)

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

                # gui 通道（乘权重 w_gui）
                if ("M_gui" not in g.columns) or (r["M_gui"] == 0):
                    e = float(r["E1_gui"])
                    log_p_gui = np.log(
                        (p_gui if e == 1.0 else 1 - p_gui) + eps)
                    log_like[d] += w_gui * log_p_gui

                # code 通道（乘权重 w_code）
                if ("M_code" not in g.columns) or (r["M_code"] == 0):
                    e = float(r["E2_code"])
                    log_p_code = np.log(
                        (p_code if e == 1.0 else 1 - p_code) + eps)
                    log_like[d] += w_code * log_p_code

                # noresp 通道（乘权重 w_no）
                if ("M_noresp" not in g.columns) or (r["M_noresp"] == 0):
                    e = float(r["E4_noresp"])
                    log_p_no = np.log((p_no if e == 1.0 else 1 - p_no) + eps)
                    log_like[d] += w_no * log_p_no

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
                    log_p_agent = np.log(psi_d + eps)
                else:
                    log_p_agent = np.log(1 - psi_d + eps)
                log_like[d] += agent_weight * log_p_agent

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

    # 从 args 获取权重参数，如果 args 为 None 则使用默认值
    w_gui = args.w_gui if args is not None else 1.0
    w_code = args.w_code if args is not None else 1.2
    w_noresp = args.w_noresp if args is not None else 0.3
    agent_weight = args.agent_weight if args is not None else 0.9
    temp = args.temp if args is not None else 1.0

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
        temp=temp,
    )

    # 加载已有参数
    em.load_params(params)
    print(f"Loaded params from {params_path}")

    post = em.predict_proba(df)

    # 构建预测结果
    pred_step = df.copy()
    pred_step["P_EnvFail"] = post[:, 0]
    pred_step["P_AgentFail"] = post[:, 1]

    # pred_correct = correct_cases_with_post(
    #     em, df, case_col="test_case_id", margin=0.0, out_dir=out_dir)
    # 从 args 获取参数，如果 args 为 None 则使用默认值
    margin_agentfail = args.margin_agentfail if args is not None else 0.8
    tau_envfail_high = args.tau_envfail_high if args is not None else 0.7
    tau_agentfail_high = args.tau_agentfail_high if args is not None else 0.95
    tau_agentfail_floor = getattr(
        args, 'tau_agentfail_floor', 0.65) if args is not None else 0.65
    tau_support = args.tau_support if args is not None else 0.0
    tau_support_pos = getattr(args, 'tau_support_pos',
                              None) if args is not None else None
    tau_support_neg = getattr(args, 'tau_support_neg',
                              None) if args is not None else None
    min_neg_channels = getattr(
        args, 'min_neg_channels', 0) if args is not None else 0

    pred_correct = correct_agent_judgment(
        df,
        em,
        margin_agentfail=margin_agentfail,
        tau_envfail_high=tau_envfail_high,
        tau_agentfail_high=tau_agentfail_high,
        tau_agentfail_floor=tau_agentfail_floor,
        tau_support=tau_support,
        tau_support_pos=tau_support_pos,
        tau_support_neg=tau_support_neg,
        min_neg_channels=min_neg_channels,
        col_case="test_case_id",
        col_agent="agent_testcase_score_x",
        out_dir=str(out_dir),
    )

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
    parser.add_argument("--data_path", type=str,
                        default="../train_em_df_webdevjudge_claude_4.xlsx")
    parser.add_argument("--test_path", type=str,
                        default="../train_em_df_webdevjudge_claude_4.xlsx")
    parser.add_argument("--params_path", type=str,
                        default="/root/tangjingyu/EM/em_adaptive_learning/src/em_adaptive_learning/em_outputs_refine_webdevjudge_claude_4/em_params.json")
    parser.add_argument("--out_dir", type=str,
                        default="em_outputs_refine_webdevjudge_claude_4")

    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--margin_agentfail", type=float, default=0.25,
                        help="Log-odds margin for flipping to AgentFail (recommended: 0.6-1.0)")
    parser.add_argument("--tau_envfail_high", type=float, default=0.72,
                        help="Threshold for flipping PASS to EnvFail")
    parser.add_argument("--tau_agentfail_high", type=float, default=1.0,
                        help="Second threshold for P_case_AgentFail when flipping to AgentFail (双门槛)")
    parser.add_argument("--tau_agentfail_floor", type=float, default=0.55,
                        help="Floor threshold for P_case_AgentFail in 0->1 flip (低保底，主门槛是 tau_support_pos)")
    parser.add_argument("--tau_support", type=float, default=0.0,
                        help="Evidence support threshold: requires avg of log-likelihood ratios >= tau_support to flip (归一化证据差异门槛，避免步数堆票)")
    parser.add_argument("--tau_support_pos", type=float, default=0.23,
                        help="Evidence support threshold for 0->1 (FAIL->PASS) flip, 相对宽松. If None, uses tau_support")
    parser.add_argument("--tau_support_neg", type=float, default=0.2,
                        help="Evidence support threshold for 1->0 (PASS->FAIL) flip, 更严格. If None, uses tau_support")
    parser.add_argument("--min_neg_channels", type=int, default=1,
                        help="Minimum number of channels supporting EnvFail for 1->0 flip (多通道一致性). 0 disables this check")

    # EM weight parameters
    parser.add_argument("--w_gui", type=float, default=0.8,
                        help="Weight for GUI evidence channel")
    parser.add_argument("--w_code", type=float, default=1.0,
                        help="Weight for code evidence channel")
    parser.add_argument("--w_noresp", type=float, default=0.05,
                        help="Weight for no-response evidence channel")
    parser.add_argument("--agent_weight", type=float, default=0.2,
                        help="Weight for agent judgment")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Temperature for softmax (1.0/1.2/1.5)")

    args = parser.parse_args()
    main(args)
    run_prediction(args.test_path, args.out_dir, args.params_path, args)
