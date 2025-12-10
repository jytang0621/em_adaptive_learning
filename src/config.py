"""
配置文件 - 包含所有路径配置
"""
from pathlib import Path

# ==================== 路径配置 ====================
# 基础路径
BASE_DIR = Path("/data/hongsirui/em_adaptive_learning")

# RealDevBench
REALDEVBENCH_DIR = BASE_DIR / "realdevbench"
# Data Directory
REALDEVBENCH_DATA_DIR = REALDEVBENCH_DIR / "data"
REALDEVBENCH_GT_JSONL = REALDEVBENCH_DATA_DIR / "realdevbench_gt.jsonl"  # gt
REALDEVBENCH_CODE_EVIDENCE_JSONL = REALDEVBENCH_DATA_DIR / "realdevbench_code_evidence.jsonl"  # code evidence
# GUI Evidence & Trajectory Directory
REALDEVBENCH_TRAJ_DIR = REALDEVBENCH_DATA_DIR / "traj"
REALDEVBENCH_GUI_EVIDENCE_JSONL = REALDEVBENCH_TRAJ_DIR / "20251120_155804" / "realdevworld_1105_res_gui_evidence.jsonl"  # gui evidence
REALDEVBENCH_PRED_FILE = REALDEVBENCH_TRAJ_DIR / "20251120_155804" / "mgx跑测_MGX低完成度_三合一_拆分case_带label_20251117_210104.xlsx"  # agent evaluation file

# WebDevJudge
WEBDEVJUDGE_DIR = BASE_DIR / "webdevjudge"
# Data Directory
WEBDEVJUDGE_DATA_DIR = WEBDEVJUDGE_DIR / "data"
WEBDEVJUDGE_GT_JSONL = WEBDEVJUDGE_DATA_DIR / "webdevjudge_unit.jsonl"  # gt
WEBDEVJUDGE_CODE_EVIDENCE_JSONL = WEBDEVJUDGE_DATA_DIR / "webdevjudge_code_evidence.jsonl"  # code evidence
# GUI Evidence & Trajectory Directory
WEBDEVJUDGE_TRAJ_DIR = WEBDEVJUDGE_DATA_DIR / "traj"
WEBDEVJUDGE_GUI_EVIDENCE_JSONL = WEBDEVJUDGE_TRAJ_DIR / "20251111_143620" / "20251111_143620_baseline_full_all_clicks.jsonl"  # gui evidence
WEBDEVJUDGE_PRED_FILE = WEBDEVJUDGE_TRAJ_DIR / "20251111_143620" / "mgx_webdevjudge_w_expected_singletest_20251111_143620.xlsx"  # agent evaluation file

