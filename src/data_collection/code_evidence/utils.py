from config import LLM_API_BASE, LLM_API_KEY, LLM_DEFAULT_MODEL
import json
import subprocess
import litellm
import re
import os
import sys

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_project_path(trace, project_root=None):
    project_name = trace['project_name']
    return project_root + '/' + project_name
    # return project_root + '/' + project_name + '/' + 'shadcn-ui'


def get_project_structure(project_path):
    """
    获取项目结构，包括所有文件和目录的路径
    """
    project_structure = []  # 存储所有文件路径

    for root, dirs, files in os.walk(project_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # 将所有文件路径添加到项目结构中
            rel_path = os.path.relpath(file_path, project_path)  # 获取相对路径
            # 排除以"."开头的隐藏文件
            if rel_path.startswith('.'):
                continue
            project_structure.append(rel_path)
    project_structure = sorted(project_structure)
    return project_structure


def call_llm(message: str, model: str = None, api_base: str = None, api_key: str = None) -> str:
    """
    调用 LLM 接口

    Args:
        message: 发送给 LLM 的消息
        model: 模型名称，默认使用 config.LLM_DEFAULT_MODEL
        api_base: API 地址，默认使用 config.LLM_API_BASE
        api_key: API 密钥，默认使用 config.LLM_API_KEY

    Returns:
        LLM 返回的文本内容
    """
    # 使用集中配置的默认值
    if model is None:
        model = LLM_DEFAULT_MODEL
    if api_base is None:
        api_base = LLM_API_BASE
    if api_key is None:
        api_key = LLM_API_KEY

    if not api_base or not api_key:
        raise ValueError(
            "API 配置缺失！请设置环境变量 LLM_API_BASE 和 LLM_API_KEY，"
            "或在 config.py 中配置"
        )

    response = litellm.completion(
        model=model,
        # temperature=0.1,
        messages=[{"role": "user", "content": message}],
        api_base=api_base,
        api_key=api_key
    )
    return response['choices'][0]['message']['content']


def match_llm_response(pattern, input_str):
    """
    使用正则表达式从输入字符串中提取匹配的内容并解析为JSON对象。
    """
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        response = match.group(1).strip()
        return json.loads(response)
    else:
        print("需要匹配的内容是\n", input_str)
        raise ValueError("未找到匹配的内容")


# 关键词搜索脚本
def search_with_script(search_term, directory=None):
    """
    使用search_dir.sh脚本搜索指定词条

    Args:
        search_term (str): 要搜索的词条
        directory (str, optional): 搜索目录，默认为当前目录

    Returns:
        dict: 包含搜索结果的字典，格式为 {'success': bool, 'results': list, 'output': str}
    """
    try:
        # 构建命令
        cmd = ['./search_dir.sh', search_term]
        if directory:
            cmd.append(directory)

        # 运行search_dir.sh脚本，设置5秒超时
        # 默认使用当前目录，可通过 directory 参数指定
        cwd = directory if directory else os.getcwd()
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                timeout=5,
                                cwd=cwd)

        # 获取搜索结果
        search_output = result.stdout

        if result.returncode == 0:
            # 解析包含<search>标签的结果
            try:
                if '<search>' in search_output and '</search>' in search_output:
                    # 提取<search>标签中的内容
                    search_pattern = r'<search>(.*?)</search>'
                    match = re.search(search_pattern, search_output, re.DOTALL)
                    if match:
                        json_content = match.group(1).strip()

                        # 使用strict=False来处理JSON中的控制字符
                        try:
                            search_results = json.loads(
                                json_content, strict=False)
                        except json.JSONDecodeError:
                            # 如果还是失败，尝试清理一些常见的控制字符
                            # 替换一些可能有问题的字符
                            cleaned_content = json_content.replace(
                                '\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                            search_results = json.loads(cleaned_content)

                        return {
                            'success': True,
                            'results': search_results,
                            'count': len(search_results)
                        }
                    else:
                        return {
                            'success': False,
                            'results': None,
                            'output': search_output,
                            'error': '无法提取<search>标签中的内容'
                        }
                else:
                    return {
                        'success': False,
                        'results': None,
                        'output': search_output,
                        'error': '搜索结果中未找到<search>标签'
                    }

            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'results': None,
                    'output': search_output,
                    'error': f'JSON解析失败: {e}'
                }
        else:
            return {
                'success': False,
                'results': None,
                'output': search_output,
                'error': f'搜索失败: {result.stderr}'
            }

    except subprocess.TimeoutExpired:
        # 超时时返回空数组
        return {
            'success': False,
            'results': [],
            'count': 0
        }
    except Exception as e:
        return {
            'success': False,
            'results': None,
            'output': '',
            'error': f'运行脚本时出错: {e}'
        }


def get_needed_files(needed_files, project_path):
    supplementary_code = {}
    for file_path in needed_files:
        full_path = os.path.join(project_path, file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                code_content = f.read()
                supplementary_code[file_path] = code_content
        except Exception as e:
            print(f"无法读取文件{full_path}: {e}")
            # raise FileNotFoundError(f"无法读取文件{full_path}: {e}")
    return supplementary_code


def save_results_to_file(results, output_path):
    # 读取现有结果
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []

    # 构建项目名称到结果的映射，方便更新
    existing_results_map = {res['project_name']                            : res for res in existing_results}

    # 更新或添加新的结果
    for res in results:
        existing_results_map[res['project_name']] = res

    # 转换回列表并写入文件
    updated_results = list(existing_results_map.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, ensure_ascii=False, indent=2)
