"""
评估工具函数
用于计算和打印准确率等评估指标
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict


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


def calculate_case_statistics(
    df: pd.DataFrame,
    case_col: str = "test_case_id",
    step_col: str = "step",
    action_col: str = "action_content",
) -> Dict:
    """
    计算case级别的统计数据，包括step数量、click动作次数、tell动作次数及其占比
    
    Args:
        df: step-level数据框
        case_col: case ID列名，默认为"test_case_id"
        step_col: step列名，默认为"step"
        action_col: action_content列名，默认为"action_content"
    
    Returns:
        包含统计结果的字典，包括case_stats DataFrame和平均值
    """
    # 先聚合到 case 级别
    case_stats = df.groupby(case_col).agg({
        step_col: "count",
        action_col: lambda x: sum(1 for v in x if pd.notna(v) and ("click(point=" in str(v) or "Run (pyautogui.click(" in str(v)))  # click 次数
    }).rename(columns={step_col: "step_count", action_col: "click_count"})
    
    # 计算 tell 次数
    tell_count = df.groupby(case_col).apply(
        lambda g: sum(1 for v in g[action_col] if pd.notna(v) and "Tell (" in str(v))
    )
    case_stats["tell_count"] = tell_count
    
    # 计算占比
    case_stats["click_ratio"] = case_stats["click_count"] / case_stats["step_count"]
    case_stats["tell_ratio"] = case_stats["tell_count"] / case_stats["step_count"]
    
    # 统计平均值
    avg_step_count = case_stats["step_count"].mean()
    avg_click_ratio = case_stats["click_ratio"].mean()
    avg_tell_ratio = case_stats["tell_ratio"].mean()
    
    print(f"Case-level statistics:")
    print(f"  Average step count per case: {avg_step_count:.2f}")
    print(f"  Average click ratio: {avg_click_ratio:.4f} ({avg_click_ratio*100:.2f}%)")
    print(f"  Average tell ratio: {avg_tell_ratio:.4f} ({avg_tell_ratio*100:.2f}%)")

    # 计算平均长度和统计信息
    print(f"Step count statistics:")
    print(f"  Mean: {case_stats['step_count'].mean():.2f}")
    print(f"  Median: {case_stats['step_count'].median():.2f}")
    print(f"  Min: {case_stats['step_count'].min()}")
    print(f"  Max: {case_stats['step_count'].max()}")
    print(f"  Std: {case_stats['step_count'].std():.2f}")
    
    return {
        "case_stats": case_stats,
        "avg_step_count": avg_step_count,
        "avg_click_ratio": avg_click_ratio,
        "avg_tell_ratio": avg_tell_ratio,
    }

