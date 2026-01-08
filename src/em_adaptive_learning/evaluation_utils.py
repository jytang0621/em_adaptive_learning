"""
评估工具函数
用于计算和打印准确率等评估指标
"""
import pandas as pd
from typing import Optional


def calculate_and_print_accuracy(
    pred_correct: pd.DataFrame,
    df: pd.DataFrame,
    gt_col: str = "phi",
    case_col: str = "test_case_id",
    prefix: str = "Gate-based correction"
) -> Optional[tuple]:
    """
    计算并打印原始准确率和矫正后的准确率
    
    Args:
        pred_correct: 包含矫正结果的DataFrame，需包含case_id, agent_original, corrected_label列
        df: step-level数据框，需包含gt_col和case_col列
        gt_col: ground truth列名，默认为"phi"
        case_col: case ID列名，默认为"test_case_id"
        prefix: 打印信息的前缀，默认为"Gate-based correction"
    
    Returns:
        如果成功计算，返回(acc_original, acc_correct)元组；否则返回None
    """
    # 如果有human_gt，计算准确率
    if gt_col not in df.columns:
        return None
    
    gt_map = df.groupby(case_col)[gt_col].first().to_dict()
    pred_correct["human_gt"] = pred_correct["case_id"].map(gt_map)
    
    valid_df = pred_correct[pred_correct["human_gt"].notna() & pred_correct["agent_original"].notna()]
    if len(valid_df) == 0:
        return None
    
    acc_original = (valid_df["agent_original"] == valid_df["human_gt"]).mean()
    acc_correct = (valid_df["corrected_label"] == valid_df["human_gt"]).mean()
    
    print(f"{prefix} - Original accuracy: {acc_original:.4f}, Corrected accuracy: {acc_correct:.4f}")
    
    return acc_original, acc_correct

