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


def convert_to_train_data(merged_df, label_col="gt_x"):
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
            "E4_noresp": agent_noresp,  # 该步“无响应 / 明显异常”等（1=有问题，0=正常）——可选，看你现有定义。
            "M_gui": M_gui,
            "M_code": M_code,
            "M_reflect": 1,
            "M_noresp": M_noresp,
            "weight": weight,
            "is_reflection": is_reflection,
            "agent_testcase_score":  agent_score
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
