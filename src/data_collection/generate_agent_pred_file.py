"""
脚本：从 /data/WebDevJudgeUnit_test 目录下的数据生成 WEBDEVJUDGE_PRED_FILE 格式的文件
生成字段：case_name, os_agent_score, evidence
"""
import os
import json
import re
import pandas as pd
from pathlib import Path


def extract_last_agent_thought(messages_data):
    """
    从 messages.json 中提取最后一步 agent 的 Thought 内容
    """
    trajectory = messages_data.get("trajectory", [])
    
    # 处理 trajectory 为 None 的情况
    if trajectory is None:
        return ""
    
    # 从后往前找最后一个 assistant 的消息
    for step in reversed(trajectory):
        if step.get("role") == "assistant":
            content = step.get("content", "")
            if isinstance(content, str):
                # 提取 Thought 部分（从 "Thought:" 开始到 "Action:" 之前）
                thought_match = re.search(r'Thought:\s*(.*?)(?:Action:|$)', content, re.DOTALL)
                if thought_match:
                    return thought_match.group(1).strip()
                # 如果没有找到 Thought: 格式，返回整个内容
                return content.strip()
    return ""


def process_webdevjudge_test_data(input_dir: str, output_file: str):
    """
    处理 WebDevJudgeUnit_test 目录下的数据，生成 pred file
    
    Args:
        input_dir: 输入目录路径，如 /data/WebDevJudgeUnit_test
        output_file: 输出文件路径，如 xxx.xlsx
    """
    input_path = Path(input_dir)
    results = []
    
    # 遍历所有 web_X 目录
    for web_dir in sorted(input_path.iterdir()):
        if not web_dir.is_dir() or not web_dir.name.startswith("web_"):
            continue
        
        # 遍历每个 web_X 下的 task_Y 目录
        for task_dir in sorted(web_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                continue
            
            messages_file = task_dir / "messages.json"
            metadata_file = task_dir / "metadata.json"
            
            if not messages_file.exists():
                print(f"警告: {messages_file} 不存在，跳过")
                continue
            
            try:
                # 读取 messages.json 获取 final_result 和 trajectory
                with open(messages_file, "r", encoding="utf-8") as f:
                    messages_data = json.load(f)
                final_result = messages_data.get("final_result", "UNKNOWN")
                
                # 读取 metadata.json 获取 web_id, task_id 和 instruction
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    web_id = metadata.get("web_id", web_dir.name)
                    task_id = metadata.get("task_id", task_dir.name.replace("task_", ""))
                    instruction = metadata.get("instruction", "")
                else:
                    web_id = web_dir.name
                    task_id = task_dir.name.replace("task_", "")
                    instruction = ""
                
                # 构造 case_name: web_X_Y 格式 (如 web_0_1, web_1_2)
                case_name = f"{web_id}_{task_id}"
                
                # 计算 os_agent_score: DONE -> 1, 其他 -> 0
                os_agent_score = 1 if final_result == "DONE" else 0
                
                # 计算 result: DONE -> Pass, 其他 -> Fail
                result = "Pass" if final_result == "DONE" else "Fail"
                
                # 提取最后一步 agent 的 Thought 作为 evidence 内容
                evidence_content = extract_last_agent_thought(messages_data)
                
                # 构造 evidence JSON 格式
                evidence = json.dumps({
                    "0": {
                        "result": result,
                        "evidence": evidence_content,
                        "case_desc": instruction
                    }
                }, ensure_ascii=False)
                
                results.append({
                    "case_name": case_name,
                    "os_agent_score": os_agent_score,
                    "evidence": evidence
                })
                
            except Exception as e:
                print(f"处理 {task_dir} 时出错: {e}")
                continue
    
    # 创建 DataFrame 并保存
    df = pd.DataFrame(results)
    
    # 按 case_name 排序 (自然排序)
    def natural_sort_key(s):
        # 提取 web_X_Y 中的 X 和 Y
        match = re.match(r'web_(\d+)_(\d+)', s)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)
    
    df['sort_key'] = df['case_name'].apply(natural_sort_key)
    df = df.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)
    
    print(f"共处理 {len(df)} 条数据")
    print(f"成功(os_agent_score=1): {len(df[df['os_agent_score'] == 1])} 条")
    print(f"失败(os_agent_score=0): {len(df[df['os_agent_score'] == 0])} 条")
    
    # 保存为 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    return df


if __name__ == "__main__":
    INPUT_DIR = "/data/WebDevJudgeUnit_test"
    OUTPUT_FILE = "/data/WebDevJudgeUnit_test/webdevjudge_pred.xlsx"
    
    df = process_webdevjudge_test_data(INPUT_DIR, OUTPUT_FILE)
    print("\n预览前3条数据:")
    for i, row in df.head(3).iterrows():
        print(f"\n=== {row['case_name']} ===")
        print(f"os_agent_score: {row['os_agent_score']}")
        print(f"evidence: {row['evidence'][:200]}...")

