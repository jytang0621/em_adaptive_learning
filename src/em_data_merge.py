import pandas as pd
import asyncio
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
)

if __name__ == "__main__":
    need_generate = False
    if need_generate:
        agent_judge = pd.read_excel(WEBDEVJUDGE_PRED_FILE)
        agent_judge = agent_judge[["case_name", "os_agent_score"]]
        agent_judge.columns = ["test_case_id", "agent_judge"]


        code_evidence = load_code_evidence(WEBDEVJUDGE_CODE_EVIDENCE_JSONL)
        code_evidence = code_evidence.merge(agent_judge, on="test_case_id", how="left")

        gui_evidence = load_gui_evidence(WEBDEVJUDGE_GUI_EVIDENCE_JSONL)
        merged_df = pd.merge(code_evidence, gui_evidence, on="test_case_id", how="left")
        merged_df_group = merged_df.groupby("test_case_id")

        
        # 调用llm生成
        filter_df = asyncio.run(filter_judge_evidence(merged_df))

        test_df = pd.merge(merged_df, filter_df, on='action_id', how='left')
        test_df.to_excel("webdevjudge_filter_df.xlsx", index=False)
    else:
        # 直接读取已经生成的数据
        test_df = pd.read_excel("train_df.xlsx")
        test_df = test_df[test_df["action_content_x"].apply(lambda x: "Stop" not in str(x))]
        test_df = test_df[test_df["action_content_x"].apply(lambda x: not pd.isna(x))]
        # train_df = sanitize_df(train_df)
    convert_to_train_data(test_df).to_excel("train_em_df_webdevjudge.xlsx")