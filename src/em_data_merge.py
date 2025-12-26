import pandas as pd
import asyncio
import numpy as np
from pathlib import Path

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
)


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


if __name__ == "__main__":
    need_generate = False
    if need_generate:
        agent_judge = pd.read_excel(WEBDEVJUDGE_PRED_FILE)
        agent_judge = agent_judge[["case_name", "os_agent_score"]]
        agent_judge.columns = ["test_case_id", "agent_judge"]

        code_evidence = load_code_evidence(WEBDEVJUDGE_CODE_EVIDENCE_JSONL)
        code_evidence = code_evidence.merge(
            agent_judge, on="test_case_id", how="left")

        gui_evidence = load_gui_evidence(WEBDEVJUDGE_GUI_EVIDENCE_JSONL)
        merged_df = pd.merge(code_evidence, gui_evidence,
                             on="test_case_id", how="left")
        merged_df_group = merged_df.groupby("test_case_id")

        # 调用llm生成
        filter_df = asyncio.run(filter_judge_evidence(merged_df))

        test_df = pd.merge(merged_df, filter_df, on='action_id', how='left')
        test_df.to_excel("webdevjudge_filter_df_claude_4.xlsx", index=False)
    else:
        # 直接读取已经生成的数据
        test_df = pd.read_excel("webdevjudge_filter_df_ui_tars.xlsx")
        test_df = test_df[test_df["action_content_x"].apply(
            lambda x: "Stop" not in str(x))]
        test_df = test_df[test_df["action_content_x"].apply(
            lambda x: not pd.isna(x))]
        # train_df = sanitize_df(train_df)
    train_em_df = convert_to_train_data(test_df, label_col="label_x",
                                        gt_file_path=GT_WEBDEVJUDGE_UNIT_JSONL)

    # 划分训练集和测试集，ratio=0.25
    train_df, val_df = make_train_val_split(train_em_df,
                                            case_col="test_case_id",
                                            val_ratio=0.25,
                                            seed=42)

    # 分别保存训练集和测试集
    train_df.to_excel(
        "em_df_webdevjudge_ui_tars_train.xlsx", index=False)
    val_df.to_excel("em_df_webdevjudge_ui_tars_test.xlsx", index=False)

    print(f"训练集大小: {len(train_df)}, 测试集大小: {len(val_df)}")
