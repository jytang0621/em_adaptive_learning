from utils import *
from PROMPT_TEST_CASE import *

# 配置文件路径 - 请根据实际情况修改
REQUIREMENTS_FILE = "./test_case_low_complete_v/requirements.json"
PROJECT_ROOT_PATH = '/path/to/your/project/workspace/'  # 请修改为实际路径
OUTPUT_DIR = '/path/to/output/'  # 请修改为实际输出路径

with open(REQUIREMENTS_FILE, "r") as f:
    requirements_data = json.load(f)

# 注意: call_llm 函数已移至 utils.py 中，使用集中配置的 API 密钥
# 请确保已设置环境变量 LLM_API_BASE 和 LLM_API_KEY

for item in requirements_data:
    project_name = item['project_name']
    print("正在处理的项目是: ", project_name)
    project_path = os.path.join(PROJECT_ROOT_PATH, project_name)
    project_structure = get_project_structure(project_path)
    # print(project_structure)

    requirements = item['requirements']

    print("第一轮调用大模型...")
    # 查找相关代码文件
    first_message = PROMPT_TEMPLATE_1.format(
        project_structure=project_structure,
        requirements=requirements
    )
    first_llm_response = call_llm(first_message)
    pattern = r'<RESPONSE>(.*?)</RESPONSE>'
    responses = match_llm_response(pattern, first_llm_response)
    print(f"第一轮调用的response是: {responses}")
    # print("第一轮调用的response是: ", len(responses))

    # 第一轮审查结束，整理补充文件和获取相应代码
    related_files = []
    for response in responses:
        for file in response['files']:
            if file not in related_files:
                related_files.append(file)
    # print("第一轮查找到的related_files: ", related_files)

    related_code = get_needed_files(related_files, project_path)
    # print("第一轮查找到的相关代码是: ", related_code)

    print("第二轮调用大模型...")
    # 生成测试用例
    third_message = PROMPT_TEMPLATE_4.format(
        requirements=requirements,
        related_code=related_code
    )
    third_llm_response = call_llm(third_message)
    pattern = r'<RESPONSE>(.*?)</RESPONSE>'
    test_cases = match_llm_response(pattern, third_llm_response)
    # print("生成的测试用例是: ", test_cases)

    save_path = os.path.join(OUTPUT_DIR, f"{project_name}.json")
    with open(save_path, "w") as f:
        json.dump(test_cases, f, indent=4, ensure_ascii=False)

    print(f"测试用例已保存到: {save_path}")
    print("---------------------------------------------------")
