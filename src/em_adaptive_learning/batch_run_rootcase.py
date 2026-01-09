"""
批量运行 run_rootcase.py 的脚本
支持在不同数据集上批量运行 EM 算法
"""
import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# 默认数据集配置
DEFAULT_DATASETS = {
    "realdevbench_claude4": {
        "data_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_claude_4_train.xlsx",
        "test_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_claude_4_test.xlsx",
        "out_dir": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_appevalpilot_claude4",
        "params_path": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_appevalpilot_claude4/em_params.json",
    },
    "webdevjudge_claude4": {
        "data_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_webdevjudge_claude_4_train.xlsx",
        "test_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_webdevjudge_claude_4_test.xlsx",
        "out_dir": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_webdevjudge_appevalpilot_claude4",
        "params_path": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_webdevjudge_appevalpilot_claude4/em_params.json",
    },
    "realdevbench_ui_tars": {
        "data_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_ui_tars_train.xlsx",
        "test_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_realdevbench_ui_tars_test.xlsx",
        "out_dir": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_ui_tars",
        "params_path": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_ui_tars/em_params.json",
    },
    "webdevjudge_ui_tars":{
        "data_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_webdevjudge_ui_tars_train.xlsx",
        "test_path": "/data/hongsirui/opensource_em_adaptive/em_adaptive_learning/src/em_df_webdevjudge_ui_tars_test.xlsx",
        "out_dir": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_webdevjudge_ui_tars",
        "params_path": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_webdevjudge_ui_tars/em_params.json",
    
    },
    # "realdevbench_claude4_query":{
    #     "data_path": "/data/hongsirui/opensource_em_adaptive/em_df_realdevbench_claude_4_train_reflection_v2_group.xlsx",
    #     "test_path": "/data/hongsirui/opensource_em_adaptive/em_df_realdevbench_claude_4_test_reflection_v2_group.xlsx",
    #     "out_dir": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_appevalpilot_claude4_query",
    #     "params_path": "/data/hongsirui/opensource_em_adaptive/em_outputs_refine_realdevbench_appevalpilot_claude4_query/em_params.json",
    # },
}


def run_single_experiment(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    val_ratio: float = 0.4,
    seed: int = 127,
    tau_agentfail: float = 1,
    tau_envfail: float = 0.65, #0.65,
    script_path: str = None,
    agent_weight: float = 0.5,
    w_gui: float = 0.8,
    w_code: float = 1.2,
    w_noresp: float = 0.3,
) -> Dict[str, Any]:
    """
    运行单个数据集的实验
    
    参数:
        dataset_name: 数据集名称
        dataset_config: 数据集配置字典
        val_ratio: 验证集比例
        seed: 随机种子
        tau_agentfail: AgentFail 阈值
        tau_envfail: EnvFail 阈值
        script_path: run_rootcase.py 的路径
    
    返回:
        包含运行结果的字典
    """
    if script_path is None:
        script_path = Path(__file__).parent / "run_rootcase.py"
    
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 构建命令
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_name", dataset_name,
        "--data_path", dataset_config["data_path"],
        "--test_path", dataset_config["test_path"],
        "--params_path", dataset_config["params_path"],
        "--out_dir", dataset_config["out_dir"],
        "--val_ratio", str(val_ratio),
        "--seed", str(seed),
        "--tau_agentfail", str(tau_agentfail),
        "--tau_envfail", str(tau_envfail),
        "--agent_weight", str(agent_weight),
        "--w_gui", str(w_gui),
        "--w_code", str(w_code),
        "--w_noresp", str(w_noresp),
    ]
    
    print(f"\n{'═'*80}")
    print(f"  🚀 Running Experiment: {dataset_name}")
    print(f"{'═'*80}")
    print(f"  📁 Data path:     {dataset_config['data_path']}")
    print(f"  🧪 Test path:    {dataset_config['test_path']}")
    print(f"  📤 Output dir:    {dataset_config['out_dir']}")
    print(f"  ⚙️  Params path:  {dataset_config['params_path']}")
    print(f"{'─'*80}")
    print(f"  ⏱️  Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*80}\n")
    
    # 运行命令并捕获输出
    em_accuracy = None
    gate_original_acc = None
    gate_corrected_acc = None
    
    try:
        # 使用Popen来同时实时显示和捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # 实时显示
            output_lines.append(line)
            
            # 解析EM准确率
            if em_accuracy is None:
                match = re.search(r'Accuracy after correction:\s*([\d.]+)', line)
                if match:
                    em_accuracy = float(match.group(1))
            
            # 解析Gate-based准确率
            if gate_original_acc is None or gate_corrected_acc is None:
                match = re.search(r'Gate-based correction - Original accuracy:\s*([\d.]+),\s*Corrected accuracy:\s*([\d.]+)', line)
                if match:
                    gate_original_acc = float(match.group(1))
                    gate_corrected_acc = float(match.group(2))
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        success = True
        error_msg = None
        full_output = ''.join(output_lines)
        
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
        full_output = ''
        print(f"\n{'─'*80}")
        print(f"  ❌ ERROR: Experiment '{dataset_name}' failed with return code {e.returncode}")
        print(f"{'─'*80}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 打印完成信息
    status_icon = "✅" if success else "❌"
    print(f"\n{'─'*80}")
    print(f"  {status_icon} Experiment '{dataset_name}' {'completed' if success else 'failed'}")
    print(f"  ⏱️  Duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"  🕐 Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*80}\n")
    
    return {
        "dataset_name": dataset_name,
        "success": success,
        "duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "error": error_msg,
        "config": dataset_config,
        "em_accuracy": em_accuracy,
        "gate_original_acc": gate_original_acc,
        "gate_corrected_acc": gate_corrected_acc,
    }


def load_config_file(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从 JSON 文件加载数据集配置
    
    参数:
        config_path: 配置文件路径
    
    返回:
        数据集配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="批量运行 run_rootcase.py，支持多个数据集"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DEFAULT_DATASETS.keys()),
        help="要运行的数据集名称列表，默认运行所有数据集",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="JSON 配置文件路径（可选，覆盖默认配置）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.5,
        help="验证集比例（默认: 0.5）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=127,
        help="随机种子（默认: 127）",
    )
    parser.add_argument(
        "--tau_agentfail",
        type=float,
        default=0.75,
        help="AgentFail 阈值（默认: 0.75）",
    )
    parser.add_argument(
        "--tau_envfail",
        type=float,
        default=0.75,
        help="EnvFail 阈值（默认: 0.75）",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help="run_rootcase.py 的路径（默认: 同目录下的 run_rootcase.py）",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default=None,
        help="保存运行摘要的 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="遇到错误时停止（默认: 继续运行下一个数据集）",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config_file:
        datasets_config = load_config_file(args.config_file)
    else:
        datasets_config = DEFAULT_DATASETS
    
    # 验证数据集名称
    invalid_datasets = [d for d in args.datasets if d not in datasets_config]
    if invalid_datasets:
        print(f"ERROR: Invalid dataset names: {invalid_datasets}")
        print(f"Available datasets: {list(datasets_config.keys())}")
        sys.exit(1)
    
    # 运行实验
    results = []
    total_start = datetime.now()
    
    print(f"\n{'█'*80}")
    print(f"  📊 BATCH RUN CONFIGURATION")
    print(f"{'█'*80}")
    print(f"  📦 Total datasets:  {len(args.datasets)}")
    print(f"  📋 Datasets:        {', '.join(args.datasets)}")
    print(f"  🔢 Val ratio:       {args.val_ratio}")
    print(f"  🎲 Seed:            {args.seed}")
    print(f"  ⚖️  Tau agentfail:   {args.tau_agentfail}")
    print(f"  ⚖️  Tau envfail:     {args.tau_envfail}")
    print(f"  🕐 Start time:      {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*80}\n")
    
    for i, dataset_name in enumerate(args.datasets, 1):
        progress = f"[{i}/{len(args.datasets)}]"
        print(f"\n{'─'*80}")
        print(f"  {progress} Processing: {dataset_name}")
        print(f"{'─'*80}")
        
        dataset_config = datasets_config[dataset_name]
        
        # 检查数据文件是否存在
        data_path = Path(dataset_config["data_path"])
        if not data_path.exists():
            print(f"  ⚠️  WARNING: Data file not found: {data_path}")
            if args.stop_on_error:
                print(f"  🛑 Stopping due to --stop_on_error flag")
                sys.exit(1)
            continue
        
        result = run_single_experiment(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            val_ratio=args.val_ratio,
            seed=args.seed,
            tau_agentfail=args.tau_agentfail,
            tau_envfail=args.tau_envfail,
            script_path=args.script_path,
        )
        
        results.append(result)
        
        if not result["success"] and args.stop_on_error:
            print(f"\n  🛑 Stopping due to error in '{dataset_name}'")
            break
    
    total_end = datetime.now()
    total_duration = (total_end - total_start).total_seconds()
    
    # 打印摘要
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count
    
    print(f"\n{'█'*80}")
    print(f"  📊 BATCH RUN SUMMARY")
    print(f"{'█'*80}")
    print(f"  ⏱️  Total duration:  {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print(f"  🕐 Start time:       {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  🕐 End time:         {total_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*80}")
    print(f"  ✅ Success:          {success_count}/{len(results)}")
    print(f"  ❌ Failed:           {failed_count}/{len(results)}")
    print(f"{'─'*80}")
    print(f"\n  📋 Detailed Results:")
    print(f"  {'─'*130}")
    print(f"  {'Status':<12} {'Dataset':<26} {'Duration':<10} {'Original':<10} {'EM Acc':<10} {'EM+Gate Acc':<12} {'Notes'}")
    print(f"  {'─'*130}")
    
    for result in results:
        status_icon = "✅" if result["success"] else "❌"
        status_text = "SUCCESS" if result["success"] else "FAILED"
        duration_str = f"{result['duration_seconds']:.2f}s"
        
        # 格式化准确率
        original_acc_str = f"{result.get('gate_original_acc', 0):.4f}" if result.get('gate_original_acc') is not None else "N/A"
        em_acc_str = f"{result.get('em_accuracy', 0):.4f}" if result.get('em_accuracy') is not None else "N/A"
        gate_acc_str = f"{result.get('gate_corrected_acc', 0):.4f}" if result.get('gate_corrected_acc') is not None else "N/A"
        
        notes = "ERROR" if not result["success"] else "OK"
        
        # 格式化数据集名称，如果太长则截断
        dataset_name = result['dataset_name']
        if len(dataset_name) > 24:
            dataset_name = dataset_name[:21] + "..."
        
        print(f"  {status_icon} {status_text:<8} {dataset_name:<26} {duration_str:<10} {original_acc_str:<10} {em_acc_str:<10} {gate_acc_str:<12} {notes}")
        if result["error"]:
            error_msg = result['error'][:75] if len(result['error']) > 75 else result['error']
            print(f"           └─ Error: {error_msg}")
    
    print(f"  {'─'*130}")
    print(f"{'█'*80}\n")
    
    # 保存摘要
    if args.output_summary:
        summary = {
            "total_duration_seconds": total_duration,
            "start_time": total_start.isoformat(),
            "end_time": total_end.isoformat(),
            "config": {
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "tau_agentfail": args.tau_agentfail,
                "tau_envfail": args.tau_envfail,
            },
            "results": results,
        }
        
        output_path = Path(args.output_summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {output_path}")
    
    # 如果有失败的实验，返回非零退出码
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

