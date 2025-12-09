from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import json
from glob import glob

from loguru import logger
from sklearn import metrics   

dir = Path(__file__).parent /  "test_cases_batch3_labels"
logger.info(dir)

def load_agent_score(gt_df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = pd.read_excel(Path(__file__).parent / "mgx_test_case_repolevel_testcase.xlsx")
    df["agent_testcase_score"] = df["test case拆分单case测试结果"].apply(lambda x: 1 if x == "pass" else 0)
    df = df.merge(gt_df, on="test_case_zh", how="left")
    df["is_correct"] = df["gt"] == df["agent_testcase_score"]
    
    df.to_excel("appevalpilot/realdevbench/testcase_generation/df.xlsx")
    agent_score_dict = {}
    group_df = df.groupby("case_name_1")
    for case_name, group in group_df:
        agent_score_dict[case_name] = {}
        testcases = group["test_case_zh"].tolist()
        agent_pred = group["agent_testcase_score"].tolist()
        agent_score_dict[case_name].update(zip(testcases,agent_pred))
    logger.info(agent_score_dict)
    return agent_score_dict

def load_code_evidence() -> tuple:
    json_files = glob(str(dir / "*.json"))
    code_evidence = {}
    count = 0
    for json_file in json_files:
        name = json_file.split("/")[-1].split(".")[0]
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _code_evidence = {}
            for item in data:
                _code_evidence[item["test_case"]] = item["code_review"]["is_implemented"]
            count += len(data)
            code_evidence[name] = _code_evidence
    logger.info(count)  
    # 展开 gt_dict 为 DataFrame，格式：case_name, test_case_zh, gt
    data_list = []
    for case_name, test_cases in code_evidence.items():
        for id, (test_case_zh, code_evidence) in enumerate(test_cases.items()):
            data_list.append({
                "case_name": case_name,
                "test_case_id": case_name + f"{id+1:02d}",
                "test_case_zh": test_case_zh,
                "code_evidence": 1 if code_evidence== True else 0
            })
    code_evidence_df = pd.DataFrame(data_list)
    return code_evidence_df

def load_gt() -> List[Dict[str, Any]]:
    json_files = glob(str(dir / "*.json"))
    gt_dict = {}
    count = 0
    for json_file in json_files:
        name = json_file.split("/")[-1].split(".")[0]
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _gt_dict = {}
            for item in data:
                _gt_dict[item["test_case"]] = item["GT"]
            count += len(data)
            gt_dict[name] = _gt_dict
    logger.info(count)
    # 展开 gt_dict 为 DataFrame，格式：case_name, test_case_zh, gt
    data_list = []
    for case_name, test_cases in gt_dict.items():
        for id, (test_case_zh, gt) in enumerate(test_cases.items()):
            data_list.append({
                "case_name": case_name,
                "test_case_id": case_name + f"{id+1:02d}",
                "test_case_zh": test_case_zh,
                "gt": gt
            })
    gt_df = pd.DataFrame(data_list)
    print(gt_df)
    return gt_dict, gt_df

def evaluate_project(gt: Dict[str, Any], agent_pred: Dict[str, Any], case_name: str = None) -> Dict[str, Any]:
    data = []
    for testcase, gt_label in gt.items():
        if testcase in agent_pred:
            agent_score = agent_pred[testcase]
            data.append({"case_name": case_name, "testcase": testcase, "gt_label": gt_label, "agent_score": agent_score})
    return data

def evaluate(gt_dict: Dict[str, Any], agent_score_dict: Dict[str, Any]) -> tuple:
    metrics_list = []
    project_names_list = []  # 跟踪成功计算的项目名称
    all_data = []  # 收集所有项目的测试用例数据
    for project_name, gt in gt_dict.items():
        agent_score = agent_score_dict[project_name]
        data = evaluate_project(gt, agent_score, case_name=project_name)
        all_data.extend(data)  # 添加到总数据中
        # logger.info(f"{project_name}:")
        try:
            metrics = get_metrics(data)
        except Exception as e:
            print(gt)
            logger.error(f"{project_name}: {e}")
            continue
        # logger.info(f"{metrics}")
        metrics_list.append(metrics)
        project_names_list.append(project_name)
    
    # 计算全部测试用例的总指标
    if all_data:
        logger.info("总体指标（全部测试用例）:")
        try:
            overall_metrics = get_metrics(all_data)
            logger.info(f"{overall_metrics}")
            metrics_list.append(overall_metrics)
            project_names_list.append("总体（全部测试用例）")
        except Exception as e:
            logger.error(f"计算总体指标时出错: {e}")
    # 增加all data输出到表格中
    all_data_df = pd.DataFrame(all_data)
    all_data_df.to_excel("appevalpilot/realdevbench/testcase_generation/all_data.xlsx")
    return metrics_list, project_names_list

def get_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt = [item["gt_label"] for item in data]
    pred = [item["agent_score"] for item in data]
    
    # 确保 confusion_matrix 总是返回 2x2 矩阵（即使只有一种类别）
    cm = metrics.confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 处理只有一种类别的情况（避免除零错误）
    precision = metrics.precision_score(gt, pred, zero_division=0)
    recall = metrics.recall_score(gt, pred, zero_division=0)
    f1 = metrics.f1_score(gt, pred, zero_division=0)
    accuracy = metrics.accuracy_score(gt, pred)
    total = len(gt)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
        "Acc": round(accuracy, 4),
        "total": total
    }

if __name__ == "__main__":
    gt_dict, gt_df = load_gt()
    agent_score_df = load_agent_score(gt_df)
    metrics_list, project_names_list = evaluate(gt_dict, agent_score_df)
    print(metrics_list)
    # print(agent_score_df.head())
    result_df = pd.DataFrame(metrics_list)
    # 为结果添加项目名称列（最后一行是总体指标）
    if len(project_names_list) == len(metrics_list):
        result_df.insert(0, "项目名称", project_names_list)
    result_df.to_excel("appevalpilot/realdevbench/testcase_generation/result.xlsx")