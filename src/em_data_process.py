import asyncio
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import json
from glob import glob
import os

from loguru import logger
from sklearn import metrics
from tqdm import tqdm
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env file loading")

from llm_provider import LLMProvider


# 从环境变量读取配置，如果没有则使用默认值
api_key = os.getenv("API_KEY", "sk-xxx")
base_url = os.getenv("BASE_URL", "https://newapi.deepwisdom.ai")
model = os.getenv("MODEL", "claude-sonnet-4-5")
provider = LLMProvider(api_key=api_key, base_url=base_url)


async def get_reflection_thought(observation):
    result = await provider.generate_reflection(reflection_thought=observation, model=model)
    # 去除 markdown 代码块标记
    result = result.strip()
    if result.startswith('```json'):
        result = result[7:]  # 去除 ```json
    elif result.startswith('```'):
        result = result[3:]  # 去除 ```
    if result.endswith('```'):
        result = result[:-3]  # 去除结尾的 ```
    result = result.strip()
    logger.info(f"result: {result}")
    try:
        return json.loads(result)['result']
    except:
        return "No"


def extract_evidence_from_tell(action_content):
    """
    从 Tell ({"0": {"result": "Pass", "evidence": "..."}}) 格式中定位并提取 evidence
    """
    if not isinstance(action_content, str) or 'Tell (' not in action_content:
        return None

    try:
        # 定位 Tell ( 后面的 JSON 对象
        evidence = action_content.split('"evidence": ')[1]
        return evidence
    except Exception as e:
        logger.error(f"Error extracting evidence: {e}")
        return None


async def filter_judge_evidence(merged_df, max_concurrent=10):
    filter_df = merged_df[merged_df['action_content'].apply(
        lambda x: isinstance(x, str) and 'Tell (' in x)]
    filter_df['agent_noresp'] = -1
    filter_df['evidence'] = None

    # 先提取所有 evidence
    for index, row in filter_df.iterrows():
        evidence = extract_evidence_from_tell(row['action_content'])
        if evidence is not None:
            filter_df.at[index, 'evidence'] = evidence

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_row(index, row, pbar):
        observation = row['evidence']

        if observation is not None:
            async with semaphore:
                try:
                    result = await get_reflection_thought(observation)
                    logger.info(
                        f"result: {result} for observation: {observation}")
                    filter_df.at[index, 'agent_noresp'] = result
                except Exception as e:
                    logger.error(f"Error processing row {index}: {e}")
                    filter_df.at[index, 'agent_noresp'] = "No"
        pbar.update(1)

    with tqdm(total=len(filter_df), desc="Filtering judge evidence") as pbar:
        tasks = [process_row(index, row, pbar)
                 for index, row in filter_df.iterrows()]
        await asyncio.gather(*tasks)

    return filter_df


def load_gt_data(gt_file_path):
    """
    加载 GT 文件并返回 GT 数据映射字典
    根据文件名自动判断匹配策略：
    - low_complete.jsonl: 使用 test_case_name 匹配，取 GT_value
    - webdevjudge_unit.jsonl: 使用 web_id + task_id 组合匹配，取 label

    Returns:
        tuple: (gt_dict, match_key, gt_value_key)
            - gt_dict: 匹配键 -> GT值 的映射字典
            - match_key: 用于匹配的键名 (test_case_name 或 test_case_id)
            - gt_value_key: GT值的字段名 (GT_value 或 label)
    """
    if gt_file_path is None:
        return None, None, None

    gt_file_path = Path(gt_file_path)
    file_name = gt_file_path.name

    # 根据文件名确定匹配策略
    if "low_complete" in file_name:
        match_key = "test_case_name"
        gt_value_key = "GT_value"
        use_composite_key = False
    elif "webdevjudge_unit" in file_name:
        # webdevjudge 使用 web_id + task_id 组合作为 key，格式为 "web_id_task_id"
        match_key = "test_case_id"  # 用于在 convert_to_train_data 中选择正确的列
        gt_value_key = "label"
        use_composite_key = True
    else:
        logger.warning(
            f"Unknown GT file type: {file_name}, using default (task_id, label)")
        match_key = "task_id"
        gt_value_key = "label"
        use_composite_key = False

    # 加载 GT 数据
    gt_dict = {}
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            value = data.get(gt_value_key)

            if use_composite_key:
                # 组合 web_id 和 task_id 生成匹配键，格式为 "web_0_1"
                web_id = data.get("web_id")
                task_id = data.get("task_id")
                if web_id is not None and task_id is not None:
                    key = f"{web_id}_{task_id}"
                else:
                    key = None
            else:
                key = data.get(match_key)

            if key is not None and value is not None:
                gt_dict[key] = value

    logger.info(
        f"Loaded {len(gt_dict)} GT entries from {file_name} using {match_key} -> {gt_value_key}")
    return gt_dict, match_key, gt_value_key


def convert_to_train_data(merged_df, label_col="gt_x", gt_file_path=None):
    """
    将合并后的 DataFrame 转换为训练数据格式

    Args:
        merged_df: 合并后的 DataFrame
        label_col: 标签列名
        gt_file_path: GT 文件路径，用于计算 delta_label
    """
    # 加载 GT 数据
    gt_dict, match_key, gt_value_key = load_gt_data(gt_file_path)

    rows_out = []
    # 1 是mask，0是不mask
    for index, row in merged_df.iterrows():
        weight = 1.0
        if row['gui_evidence_x'] == -1:
            M_gui = 1
            gui_evidence = 0
            weight = 1
        else:
            M_gui = 0
            gui_evidence = row['gui_evidence_x']

        # agent_testcase_score 始终从 agent_judge_x 取值（即 os_agent_score）
        agent_testcase_score_val = row['agent_judge_x']

        if row["agent_noresp"] is np.nan:
            M_reflect = 1
            M_noresp = 1
            agent_noresp = 0
            # M_gui = 1
            M_code = 0
            weight = 0.3
            agent_score = np.nan
            is_reflection = 0
        else:
            M_reflect = 1
            M_noresp = 0
            M_gui = 1
            M_code = 0
            weight = 0.5
            agent_noresp = 0 if row["agent_noresp"] == "Yes" else 1
            agent_score = row['agent_judge_x']
            is_reflection = 1

        # 计算 delta_label（使用 agent_testcase_score_val 而非 agent_score）
        delta_label = np.nan  # 默认值
        if gt_dict is not None:
            # 根据 match_key 选择正确的匹配列
            # - webdevjudge_unit.jsonl (match_key="test_case_id"): 使用 test_case_id_x 列
            # - low_complete.jsonl (match_key="test_case_name"): 使用 test_case_id_x 列
            lookup_key = row['test_case_id_x']

            gt_value = gt_dict.get(lookup_key)
            if gt_value is not None and not pd.isna(agent_testcase_score_val):
                # 比较 GT_value 和 agent_judge_x
                # 相同则 delta_label = 0，不同则 delta_label = 1
                logger.info(
                    f"gt_value: {gt_value}, agent_score: {agent_testcase_score_val}")
                delta_label = 0 if gt_value == agent_testcase_score_val else 1

        rows_out.append({
            "test_case_id": row['test_case_id_x'],
            "step": row['action_id'],
            "phi": row[label_col],
            "operation_desc": row['operation_desc_x'],
            "action_content": row['action_content_x'],
            "E1_gui": gui_evidence,  # 该步 GUI 点击是否符合预期（1=好，0=坏）。
            "E2_code": row['code_evidence_x'],  # 该步代码审查是否符合预期（1=好，0=坏）。
            # E3_reflect：仅在 is_reflection=1 的行有值，用于建模 H（truthful vs hallucination）。
            "E3_reflect": agent_score,
            "E4_noresp": agent_noresp,  # 该步"无响应 / 明显异常"等（1=有问题，0=正常）——可选，看你现有定义。
            "M_gui": M_gui,
            "M_code": M_code,
            "M_reflect": 1,
            "M_noresp": M_noresp,
            "weight": weight,
            "is_reflection": is_reflection,
            "agent_testcase_score": agent_testcase_score_val,  # 始终填入 os_agent_score
            "delta_label": delta_label  # 新增：GT 与 agent_judge 的差异标签
        })

    return pd.DataFrame(rows_out)


if __name__ == "__main__":
    pass
    # train_df = pd.read_excel("train_df.xlsx")
    # train_df = train_df[train_df["action_content_x"].apply(lambda x: "Stop" not in str(x))]
    # train_df = train_df[train_df["action_content_x"].apply(lambda x: not pd.isna(x))]
    # train_df.to_excel("train_df.xlsx")

    # train_df = sanitize_df(train_df)
    # convert_to_train_data(train_df).to_excel("train_em_df.xlsx")
    # '''
