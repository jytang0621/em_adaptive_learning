from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import json
from glob import glob

from loguru import logger
from sklearn import metrics

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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def get_gui_evidence(traj):
    gui_evidence = []
    for action in traj:
        if action['coordinate_analysis'] is not None:
            try:
                gui_score = action['coordinate_analysis']['accuracy']
                gui_evidence.append(gui_score)
            except:
                gui_evidence.append(0) # 0 means fail
        else:
            gui_evidence.append(-1) # -1 means 不需要进行分析的动作
    return gui_evidence

def load_gui_evidence(path: Path) -> pd.DataFrame:
    """统一的函数，用于加载 GUI evidence 数据，自动检测数据格式"""
    gui_evidence = []
    for row in load_jsonl(path):
        project_name = row["project_name"]
        # 检测数据格式：如果包含 'results' 字段，说明是列表格式
        actions = row["results"] if "results" in row else [row]
        gui_evidence_list = get_gui_evidence(actions)
        
        for i, action in enumerate(actions):
            iter_num = i + 1 if "results" in row else row["iter_num"]
            gui_evidence.append({
                "test_case_id": project_name,
                "action_id": f"{project_name}_iter_{iter_num}",
                "operation_desc": action["operation_desc"],
                "reflection_thought": action["reflection_thought"],
                "action_content": action["action_content"],
                "gui_evidence": gui_evidence_list[i]
            })
    return pd.DataFrame(gui_evidence)


def load_code_evidence(path: Path, tag="webdevjudge") -> pd.DataFrame:
    rows = load_jsonl(path)
    
    for row in rows:
        if tag == "webdevjudge":
            row.pop("code", None)
            row["test_case_id"] = f"{row['web_id']}_{row['task_id']}"
            row["code_evidence"] = int(row["code_review"]["is_implemented"])
            row["code_evidece_reason"] = row["code_review"]["evidence"]
        else:  # realdevbench
            row["test_case_id"] = row['test_case_name']
            row["code_evidence"] = 1
            row["code_evidece_reason"] = row["Code_Evidence"]
    
    # 计算准确率
    if tag == "webdevjudge":
        acc = sum(row["label"] == row["code_review"]["is_implemented"] for row in rows)
    else:  # realdevbench
        acc = sum(row["GT"] == row["Code_Result"] for row in rows)
    
    print(f"accuracy: {acc / len(rows)}")
    return pd.DataFrame(rows)

def load_gt(path: Path, tag="webdevjudge") -> pd.DataFrame:
    """统一的函数，用于加载 GT 数据，根据 tag 自动处理不同格式"""
    rows = load_jsonl(path)
    
    for row in rows:
        if tag == "webdevjudge":
            row.pop("code", None)
            row["case_name"] = f"{row['web_id']}_{row['task_id']}"
    
    gt_df = pd.DataFrame(rows)
    
    # realdevbench 格式需要重命名列
    if tag == "realdevbench":
        gt_df = gt_df.rename(columns={"test_case_name": "case_name", "GT_value": "label"}, errors="ignore")
    
    return gt_df



def evaluate_project(gt: Dict[str, Any], agent_pred: Dict[str, Any], case_name: str = None) -> Dict[str, Any]:
    data = []
    for testcase, gt_label in gt.items():
        if testcase in agent_pred:
            agent_score = agent_pred[testcase]
            data.append({"case_name": case_name, "testcase": testcase, "gt_label": gt_label, "agent_score": agent_score})
    return data

def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, Any]:
    gt = gt_df["label"].astype(int).tolist()
    pred = gt_df["os_agent_score"].astype(int).tolist()

    tn, fp, fn, tp = metrics.confusion_matrix(gt, pred).ravel()
    precision = metrics.precision_score(gt, pred)
    recall = metrics.recall_score(gt, pred)
    f1 = metrics.f1_score(gt, pred)
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


def process_multitest(pred_df: pd.DataFrame) -> pd.DataFrame:
    case_name_items = []
    os_agent_score_items = []
    evidence_items = []

    evidence = pred_df["evidence"].tolist()
    case_names = pred_df["case_name"].tolist()

    for idx, info in enumerate(evidence):
        items  = json.loads(info)
        for key, value in items.items():
            case_name_items.append(case_names[idx]+"_"+str(int(key)+1))
            os_agent_score_items.append(1 if value["result"]=="Pass" else 0)
            evidence_items.append(value["evidence"])
    
    df = pd.DataFrame({"case_name": case_name_items, "os_agent_score": os_agent_score_items, "evidence": evidence_items})
    return df
    
def run_evaluation(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tag="wo_expected + singletest", is_batch=False) -> pd.DataFrame:
    if is_batch:
        pred_df = process_multitest(pred_df)
    gt_df_merged = gt_df.merge(pred_df, on="case_name", how="left")#label是gt，os_agent_score是pred
    gt_df_merged = gt_df_merged[gt_df_merged["os_agent_score"].notna()]
    result_df = evaluate(gt_df_merged, pred_df)
    print(tag)
    print(result_df)
    return result_df

def load_pred(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_excel(path)
    df = df[["case_name", "os_agent_score", "evidence"]]
    return df

def run_evaluation_realdevbench(gt_file: Path, pred_file: Path) -> pd.DataFrame:
    gt_df = load_gt(gt_file, tag="realdevbench")
    pred_df = load_pred(pred_file)
    result_df = run_evaluation(gt_df, pred_df, tag="realdevbench_low_complete_singletest", is_batch=False)
    return result_df

def run_evaluation_webdevjudge(gt_file: Path, pred_file: Path) -> pd.DataFrame:
    gt_df = load_gt(gt_file, tag="webdevjudge")
    pred_df = load_pred(pred_file)
    result_df = run_evaluation(gt_df, pred_df, tag="w_expected + singletest_mini", is_batch=False)
    return result_df

if __name__ == "__main__":
    # 测试加载代码证据
    load_code_evidence(REALDEVBENCH_CODE_EVIDENCE_JSONL, tag="realdevbench")
    load_code_evidence(WEBDEVJUDGE_CODE_EVIDENCE_JSONL, tag="webdevjudge")

    # 测试加载GUI证据
    gui_evidence = load_gui_evidence(WEBDEVJUDGE_GUI_EVIDENCE_JSONL)
    print(gui_evidence.head(3))
    gui_evidence = load_gui_evidence(REALDEVBENCH_GUI_EVIDENCE_JSONL)
    print(gui_evidence.head(3))

    # 测试评估
    run_evaluation_realdevbench(REALDEVBENCH_GT_JSONL, REALDEVBENCH_PRED_FILE)
    run_evaluation_webdevjudge(WEBDEVJUDGE_GT_JSONL, WEBDEVJUDGE_PRED_FILE)