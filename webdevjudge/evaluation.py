from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import json

from sklearn import metrics
dir = Path(__file__).parent 

def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, Any]:
    gt = gt_df["label"].astype(int).tolist()
    pred = gt_df["os_agent_score"].astype(int).tolist()
    # print(metrics.classification_report(gt, pred))
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

def load_gt_realdevbench(path: Path) -> List[Dict[str, Any]]:
    info = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            info.append(data)
    gt_df = pd.DataFrame(info)
    gt_df = gt_df.rename(columns={"test_case_name": "case_name",
                                    "GT_value": "label"})
    return gt_df

def load_gt(path: Path) -> List[Dict[str, Any]]:
    info = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data.pop("code")
            data["case_name"] = data["web_id"]+"_"+str(data["task_id"])
            info.append(data)
    gt_df = pd.DataFrame(info)
    return gt_df

def load_pred(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_excel(path)
    df = df[["case_name", "os_agent_score", "evidence"]]
    return df

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

def run_evaluation_realdevbench() -> pd.DataFrame:
    # realdevbench 低完成度
    realdevbench_low_complete_dir = dir.parent / "realdevbench" 
    gt_df = load_gt_realdevbench(dir / "realdevbench_low_complete.jsonl")
    # claude4-0514
    # pred_df = load_pred(realdevbench_low_complete_dir /  "traj" / "20251120_155804_c4s" / "mgx跑测_MGX低完成度_三合一_拆分case_带label_20251120_155804_c4s_0514（重跑）.xlsx")
    # claude3-5
    pred_df = load_pred(realdevbench_low_complete_dir /  "traj" / "20251120_155804_c4s" / "mgx跑测_MGX低完成度_三合一_拆分case_带label_20251117_210104.xlsx")
    print(gt_df.head())
    print(pred_df.head())
    result_df = run_evaluation(gt_df, pred_df, tag="realdevbench_low_complete_singletest", is_batch=False)

def run_evaluation_webdevjudge() -> pd.DataFrame:
    gt_df = load_gt(dir / "webdevjudge_unit.jsonl")
    result_df = run_evaluation(gt_df, pred_df, tag="w_expected + singletest_mini", is_batch=False)
    pred_df = load_pred(dir / "traj" / "20251121_170336" / "mgx跑测_WebDevJudge_三合一_拆分case_带label_20251121_170336_c4s.xlsx")
    result_df = run_evaluation(gt_df, pred_df, tag="w_expected + singletest_mini", is_batch=False)

if __name__ == "__main__":
    # realdevbench 低完成度
    result_df = run_evaluation_realdevbench()
    print(result_df)

    # webdevjudge 单元测试
    # result_df = run_evaluation_webdevjudge()
    # print(result_df)

'''
    # w_expected + singletest
    pred_df_w_expected_singletest = load_pred(dir / "mgx_webdevjudge_w_expected_singletest.xlsx")
    valid_test_cases = pred_df_w_expected_singletest["case_name"].tolist()
    # gt_df = gt_df[gt_df["case_name"].isin(valid_test_cases)]

    # wo_expected + singletest
    pred_df_wo_expected_singletest = load_pred(dir / "mgx_webdevjudge_wo_expected_singletest.xlsx")
    
    # wo_expected + multitest
    pred_df_wo_expected_multitest = load_pred(dir / "mgx_webdevjudge_wo_expected_multitest.xlsx")
    # pred_df_wo_expected_multitest = process_multitest(pred_df_wo_expected_multitest)
    # w_expected + multitest
    pred_df_w_expected_multitest = load_pred(dir / "mgx_webdevjudge_w_expected_multitest.xlsx")
    # pred_df_w_expected_multitest = process_multitest(pred_df_w_expected_multitest)

    # 1105 分离版本 + 带expected + 逐个测试
    pred_df_w_expected_singletest_dev = load_pred(dir / "mgx_webdevjudge_w_expected_singletest_dev.xlsx")

    result_wo_expected_singletest = run_evaluation(gt_df, pred_df_wo_expected_singletest, tag="wo_expected + singletest", is_batch=False)
    result_w_expected_singletest = run_evaluation(gt_df, pred_df_w_expected_singletest, tag="w_expected + singletest", is_batch=False)
    result_wo_expected_multitest = run_evaluation(gt_df, pred_df_wo_expected_multitest, tag="wo_expected + multitest", is_batch=True)
    result_w_expected_multitest = run_evaluation(gt_df, pred_df_w_expected_multitest, tag="w_expected + multitest", is_batch=True)


    print(len(gt_df))
    result_w_expected_singletest_dev = run_evaluation(gt_df, pred_df_w_expected_singletest_dev, tag="w_expected + singletest dev", is_batch=False)
'''   