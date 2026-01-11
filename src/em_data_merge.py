import pandas as pd
import asyncio
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, Any, Set

from sklearn import metrics
from evaluation import load_code_evidence, load_gui_evidence
from em_data_process import filter_judge_evidence, convert_to_train_data
# 从配置文件导入路径配置
from config import (
    BASE_DIR,
    REALDEVBENCH_DIR,
    REALDEVBENCH_DATA_DIR,
    REALDEVBENCH_GT_JSONL,
    REALDEVBENCH_CODE_EVIDENCE_JSONL,
    REALDEVBENCH_TRAJ_DIR,
    REALDEVBENCH_GUI_EVIDENCE_JSONL,
    REALDEVBENCH_PRED_FILE,
    WEBDEVJUDGE_DIR,
    WEBDEVJUDGE_DATA_DIR,
    WEBDEVJUDGE_GT_JSONL,
    WEBDEVJUDGE_CODE_EVIDENCE_JSONL,
    WEBDEVJUDGE_TRAJ_DIR,
    WEBDEVJUDGE_GUI_EVIDENCE_JSONL,
    WEBDEVJUDGE_PRED_FILE,
    GT_WEBDEVJUDGE_UNIT_JSONL,
    GT_LOW_COMPLETE_JSONL,
)


def make_train_val_split(df: pd.DataFrame,
                         case_col: str = "test_case_id",
                         val_ratio: float = 0.2,
                         seed: int = 42,
                         train_ref_file: Path = None,
                         val_ref_file: Path = None):
    """
    按 case 粒度划分 train/val

    Args:
        df: 待划分的数据
        case_col: case id 列名
        val_ratio: 验证集比例（仅在随机划分时使用）
        seed: 随机种子（仅在随机划分时使用）
        train_ref_file: 训练集参考文件路径，包含 case_col 列
        val_ref_file: 验证集参考文件路径，包含 case_col 列

    Returns:
        (train_df, val_df) 元组
    """
    if train_ref_file is not None and val_ref_file is not None:
        # 从参考文件读取 case_id 集合
        train_ref_df = pd.read_excel(train_ref_file)
        val_ref_df = pd.read_excel(val_ref_file)
        train_ids = set(train_ref_df[case_col].astype(str).unique())
        val_ids = set(val_ref_df[case_col].astype(str).unique())
        print(
            f"使用参考文件划分: train={len(train_ids)} cases, val={len(val_ids)} cases")
    else:
        # 原有随机划分逻辑
        rng = np.random.default_rng(seed)
        case_ids = df[case_col].astype(str).unique()
        rng.shuffle(case_ids)
        n_val = int(len(case_ids) * val_ratio)
        val_ids = set(case_ids[:n_val])
        train_ids = set(case_ids[n_val:])
        print(f"使用随机划分 (seed={seed}, val_ratio={val_ratio}): "
              f"train={len(train_ids)} cases, val={len(val_ids)} cases")

    train_df = df[df[case_col].astype(str).isin(
        train_ids)].reset_index(drop=True)
    val_df = df[df[case_col].astype(str).isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def _load_gt_labels(gt_file: Path) -> Dict[str, int]:
    """加载 GT 文件，返回 case_name -> label 的映射"""
    gt_map = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_name = f"{row['web_id']}_{row['task_id']}"
            gt_map[case_name] = int(row['label'])
    return gt_map


def _compute_metrics(gt_labels: list, pred_labels: list) -> Dict[str, Any]:
    """计算准确率相关指标"""
    if len(gt_labels) == 0:
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0,
                "P": 0.0, "R": 0.0, "F1": 0.0, "Acc": 0.0, "total": 0}

    tn, fp, fn, tp = metrics.confusion_matrix(
        gt_labels, pred_labels, labels=[0, 1]).ravel()
    precision = metrics.precision_score(
        gt_labels, pred_labels, zero_division=0)
    recall = metrics.recall_score(gt_labels, pred_labels, zero_division=0)
    f1 = metrics.f1_score(gt_labels, pred_labels, zero_division=0)
    accuracy = metrics.accuracy_score(gt_labels, pred_labels)

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
        "Acc": round(accuracy, 4),
        "total": len(gt_labels)
    }


def _get_case_last_row(df: pd.DataFrame, case_col: str = "test_case_id") -> pd.DataFrame:
    """
    按 case 分组，取每个 case 的最后一行

    Args:
        df: 输入 DataFrame
        case_col: case 列名

    Returns:
        每个 case 最后一行组成的 DataFrame
    """
    return df.groupby(case_col).last().reset_index()


def evaluate_split_accuracy(
    pred_file: Path,
    train_split_file: Path,
    val_split_file: Path,
    gt_file: Path,
    case_col: str = "test_case_id",
    pred_case_col: str = "case_name",
    pred_score_col: str = "os_agent_score"
) -> Dict[str, Dict]:
    """
    根据已有的训练集/测试集划分，对 pred 文件切分并计算准确率

    Args:
        pred_file: 预测文件路径 (xlsx)，包含 case_name 和 os_agent_score 列
        train_split_file: 训练集划分文件路径 (xlsx)，包含 test_case_id 列
        val_split_file: 测试集划分文件路径 (xlsx)，包含 test_case_id 列
        gt_file: GT 文件路径 (jsonl)，包含 web_id, task_id, label 字段
        case_col: 划分文件中的 case 列名
        pred_case_col: 预测文件中的 case 列名
        pred_score_col: 预测文件中的预测分数列名

    Returns:
        {"train": {...metrics...}, "val": {...metrics...}}
    """
    # 1. 从 train/val split 文件读取 unique 的 case_id 集合
    train_df = pd.read_excel(train_split_file)
    val_df = pd.read_excel(val_split_file)
    train_case_ids: Set[str] = set(train_df[case_col].astype(str).unique())
    val_case_ids: Set[str] = set(val_df[case_col].astype(str).unique())

    # 2. 加载 pred 文件
    pred_df = pd.read_excel(pred_file)
    pred_df[pred_case_col] = pred_df[pred_case_col].astype(str)

    # 3. 加载 GT 文件
    gt_map = _load_gt_labels(gt_file)

    # 4. 根据 case_id 切分 pred 数据并计算指标
    results = {}

    for split_name, case_ids in [("train", train_case_ids), ("val", val_case_ids)]:
        # 筛选属于当前 split 的预测数据
        split_pred = pred_df[pred_df[pred_case_col].isin(case_ids)]

        gt_labels = []
        pred_labels = []

        for _, row in split_pred.iterrows():
            case_name = str(row[pred_case_col])
            if case_name in gt_map:
                gt_labels.append(gt_map[case_name])
                pred_labels.append(int(row[pred_score_col]))

        results[split_name] = _compute_metrics(gt_labels, pred_labels)
        print(f"{split_name.upper()} Set - Cases: {len(case_ids)}, "
              f"Matched: {len(gt_labels)}, Metrics: {results[split_name]}")

    return results


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    need_generate = True

    # Delta Label 标注模式配置（三选一，互斥）:
    # - "default": 现有逻辑（gt=1且agent=0→1，gt=0且agent=1→0）
    # - "extend_negative": 在 default 基础上，按 delta_ratio 比例对 gt=0且agent=0 标注为 0
    # - "reduce_positive": 按 delta_ratio 比例保留 delta_label=1，其余置为 np.nan
    # 可选: "default", "extend_negative", "reduce_positive"
    delta_mode = "extend_negative"
    delta_ratio = 1       # 比例参数（仅 extend_negative 和 reduce_positive 模式生效）
    delta_seed = 42         # 随机种子

    if need_generate:
        agent_judge = pd.read_excel(REALDEVBENCH_PRED_FILE)
        agent_judge = agent_judge[["case_name", "os_agent_score"]]
        # 兼容 case_name 格式: 去掉末尾的 _数字 后缀 (如 xxx_case365_365 -> xxx_case365)
        agent_judge["case_name"] = agent_judge["case_name"].apply(
            lambda x: re.sub(r'_\d+$', '', str(x))
        )
        agent_judge.columns = ["test_case_id", "agent_judge"]

        code_evidence = load_code_evidence(
            REALDEVBENCH_CODE_EVIDENCE_JSONL, tag="realdevbench")
        code_evidence = code_evidence.merge(
            agent_judge, on="test_case_id", how="left")

        gui_evidence = load_gui_evidence(
            REALDEVBENCH_GUI_EVIDENCE_JSONL, tag="realdevbench")
        merged_df = pd.merge(code_evidence, gui_evidence,
                             on="test_case_id", how="left")
        merged_df_group = merged_df.groupby("test_case_id")

        # 调用llm生成
        filter_df = asyncio.run(filter_judge_evidence(merged_df))

        test_df = pd.merge(merged_df, filter_df, on='action_id', how='left')
        test_df.to_excel(
            "realdevbench_filter_df_ui_tars.xlsx", index=False)
    else:
        # 直接读取已经生成的数据
        test_df = pd.read_excel(
            "realdevbench_filter_df_ui_tars.xlsx")
        # 不再过滤 Stop 和 NaN 的行，对于没有 Tell/Stop 操作的行，agent_noresp 保持 NaN
        # filter_judge_evidence 只处理包含 'Tell (' 的行，其他行 merge 后 agent_noresp 会是 NaN
        # train_df = sanitize_df(train_df)
    label_col = "GT_x" if "GT_x" in test_df.columns else "label_x"
    train_em_df = convert_to_train_data(
        test_df,
        label_col=label_col,
        gt_file_path=GT_LOW_COMPLETE_JSONL,
        delta_mode=delta_mode,
        delta_ratio=delta_ratio,
        seed=delta_seed
    )

    print(
        f"\n使用 delta_mode='{delta_mode}', delta_ratio={delta_ratio}, seed={delta_seed}")

    # ==================== 整个数据集的准确率（按 case 计算） ====================
    print("\n" + "=" * 60)
    print("整个数据集的准确率（按 case，取最后一行）")
    print("=" * 60)

    # 按 case 取最后一行
    all_case_last = _get_case_last_row(train_em_df, case_col="test_case_id")

    # Code Evidence 与 GT 的准确率
    all_code_metrics = _compute_metrics(
        all_case_last["phi"].astype(int).tolist(),
        all_case_last["E2_code"].astype(int).tolist()
    )
    print(
        f"Code Evidence vs GT - Acc: {all_code_metrics['Acc']}, Metrics: {all_code_metrics}")

    # Agent 预测与 GT 的准确率
    all_case_last_valid = all_case_last[all_case_last["agent_testcase_score"].notna(
    )]
    all_agent_metrics = _compute_metrics(
        all_case_last_valid["phi"].astype(int).tolist(),
        all_case_last_valid["agent_testcase_score"].astype(int).tolist()
    )
    print(
        f"Agent vs GT - Acc: {all_agent_metrics['Acc']}, Metrics: {all_agent_metrics}")

    # 划分训练集和测试集，ratio=0.25
    # True: 使用参考文件切分; False: 随机切分
    use_ref_split = True

    if use_ref_split:
        train_df, val_df = make_train_val_split(
            train_em_df,
            case_col="test_case_id",
            val_ratio=0.25,
            train_ref_file=Path("em_df_realdevbench_claude_4_train.xlsx"),
            val_ref_file=Path("em_df_realdevbench_claude_4_test.xlsx"),
        )
    else:
        train_df, val_df = make_train_val_split(
            train_em_df,
            case_col="test_case_id",
            val_ratio=0.25,
            seed=1010
        )

    # 分别保存训练集和测试集
    train_df.to_excel(
        "em_df_realdevbench_ui_tars_train_delta_extend_negative.xlsx", index=False)
    val_df.to_excel(
        "em_df_realdevbench_ui_tars_test_delta_extend_negative.xlsx", index=False)

    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(val_df)}")

    # 计算训练集和测试集上 code_evidence 与 GT 的准确率（按 case）
    print("\n" + "=" * 60)
    print("Code Evidence 与 GT 准确率（按子集，按 case 取最后一行）")
    print("=" * 60)

    # 按 case 取最后一行
    train_case_last = _get_case_last_row(train_df, case_col="test_case_id")
    val_case_last = _get_case_last_row(val_df, case_col="test_case_id")

    train_metrics = _compute_metrics(
        train_case_last["phi"].astype(int).tolist(),
        train_case_last["E2_code"].astype(int).tolist()
    )
    print(f"Train Set - Acc: {train_metrics['Acc']}, Metrics: {train_metrics}")

    val_metrics = _compute_metrics(
        val_case_last["phi"].astype(int).tolist(),
        val_case_last["E2_code"].astype(int).tolist()
    )
    print(f"Val Set - Acc: {val_metrics['Acc']}, Metrics: {val_metrics}")

    # ==================== Agent 预测与 GT 准确率（按子集） ====================
    print("\n" + "=" * 60)
    print("Agent 预测与 GT 准确率（按子集，按 case 取最后一行）")
    print("=" * 60)

    # 过滤掉 agent_testcase_score 为 NaN 的行（使用已按 case 聚合的数据）
    train_case_last_valid = train_case_last[train_case_last["agent_testcase_score"].notna(
    )]
    val_case_last_valid = val_case_last[val_case_last["agent_testcase_score"].notna(
    )]

    train_agent_metrics = _compute_metrics(
        train_case_last_valid["phi"].astype(int).tolist(),
        train_case_last_valid["agent_testcase_score"].astype(int).tolist()
    )
    print(
        f"Train Set - Acc: {train_agent_metrics['Acc']}, Metrics: {train_agent_metrics}")

    val_agent_metrics = _compute_metrics(
        val_case_last_valid["phi"].astype(int).tolist(),
        val_case_last_valid["agent_testcase_score"].astype(int).tolist()
    )
    print(
        f"Val Set - Acc: {val_agent_metrics['Acc']}, Metrics: {val_agent_metrics}")
