# Data Collection 模块

本模块用于从 Agent 执行轨迹中提取数据，供 EM 自适应学习模块 (`em_adaptive_learning/`) 进行根因分析。

## 📁 文件结构

```
data_collection/
├── README.md                          # 本文档
├── generate_agent_pred_file.py        # 生成 Agent 预测结果文件 (agent_pred.xlsx)
├── gui_evidence/                      # GUI Evidence 生成器
│   ├── gui_evidence_generator.py      # AppEvalPilot Agent 专用 (有元素树信息)
│   ├── ui_tars_evidence_generator.py  # 其他 Agent 通用 (UI-TARS 等，使用 OCR)
│   └── screenshot2info.py             # 截图信息提取工具 (OCR + 图标检测)
└── code_evidence/                     # Code Evidence 生成器
    ├── test_case_generation.py        # 测试用例生成主程序
    ├── utils.py                       # 工具函数 (LLM 调用、文件处理等)
    └── PROMPT_TEST_CASE.py            # LLM 提示词模板
```

---

## ⚙️ API 配置

本模块的 LLM 调用使用集中配置的 API 密钥，配置位于 `src/config.py`。

**配置方式 (推荐使用环境变量)**：

```bash
# 设置环境变量
export LLM_API_BASE="https://your-api-endpoint.com"
export LLM_API_KEY="your-api-key-here"
export LLM_DEFAULT_MODEL="claude-sonnet-4-20250514" 
```

---

## 🤖 支持的 Agent 类型

| Agent 类型 | 生成器 | 数据来源 | 元素定位方式 |
|------------|--------|----------|--------------|
| **AppEvalPilot** | `gui_evidence_generator.py` | info.txt (含元素树) | 元素树匹配 (精确) |
| **UI-TARS** | `webvoyager_evidence_generator.py` | messages.json + screenshots | OCR + 图标检测 |

> 💡 我们以 **UI-TARS** ([WebDevJudge](https://github.com/lcy2723/WebDevJudge) 中使用的 agent 框架，结合 UI-TARS) 为例，验证了对非 AppEvalPilot Agent 的支持。只要 Agent 输出符合 `messages.json` + `screenshots/` 的结构，均可使用 `webvoyager_evidence_generator.py` 处理。

---

## 📄 生成文件格式说明

### 1. `gui_evidence.jsonl` - GUI Evidence 数据

每行是一个 JSON 对象，代表一个完整的任务 (task/project)。

```jsonc
{
  "project_name": "web_0_1",           // 项目/任务唯一标识
  "project_folder": "/data/xxx",       // 原始数据路径
  "status": "success",                 // 处理状态: success/error
  "error": null,                       // 错误信息 (如有)
  "processed_at": "2024-12-14T10:00:00", // 处理时间
  
  // 每一步迭代的结果
  "results": [
    {
      "iter_num": 1,                   // 迭代编号
      "reflection": null,              // 反思结果: 1=正确, 0=错误, null=无反思
      "click_coords": [450, 722],      // 点击坐标 [x, y]
      "operation_desc": "点击搜索按钮", // 操作描述
      "action_content": "click(...)",  // 执行的动作内容
      "action_type": "click",          // 动作类型: click/type/scroll/drag/hotkey/wait/finished
      "action_target": "(450, 722)",   // 动作目标描述
      "reflection_thought": "...",     // 反思思考内容
      
      // 坐标匹配结果 (用于 E1_gui 证据)
      "coordinate_match": 1,           // 1=命中, 0=未命中, null=无分析
      "coordinate_analysis": {
        "accuracy": 1,
        "method": "element_tree",      // 分析方法: element_tree/mllm/ocr
        "matched_element_id": "element_0"
      },
      
      // 🔑 关键字段: 元素距离排序 (按距离从近到远)
      "element_distance_sorting": [
        {
          "id": "element_0",
          "ui_name": "搜索",           // 元素文本/名称
          "control_type": "Button",    // 控件类型
          "bbox": [400, 700, 500, 750], // 边界框 [x1, y1, x2, y2]
          "distance": 35.5,            // 到点击坐标的距离
          "area": 5000,                // 元素面积
          "is_inside": true            // 点击点是否在元素内部
        },
        // ... 更多元素 (按距离从近到远排序)
      ],
      
      // 最后一步特有字段
      "final_result": "DONE"           // 任务最终结果: DONE/FAILED/INITIAL_ERROR
    }
    // ... 更多迭代
  ],
  
  // 摘要统计
  "summary": {
    "total_iters": 5,                  // 总迭代数
    "reflection_found": 3,             // 找到反思结果的数量
    "reflection_error_cases": 1,       // 反思判定为错误的数量
    "coordinate_analyzed": 4           // 进行坐标分析的数量
  },
  
  // 坐标分析统计
  "coordinate_analysis": {
    "total_coordinate_cases": 4,
    "successfully_analyzed": 4,
    "coordinate_analysis_dir": "/data/xxx/coordinate_analysis",
    "coordinate_files_created": 4,
    "element_tree_cases": 4,
    "element_tree_matched": 3,
    "mllm_cases": 0
  }
}
```

**关键字段用途**：

| 字段 | 用途 | 说明 |
|------|------|------|
| `element_distance_sorting` | 生成 **E1_gui** 证据 | 判断点击是否命中目标元素 |
| `coordinate_match` | E1_gui 二值证据 | 1=命中, 0=未命中 |
| `action_content` | 分析操作意图 | 用于生成 E2_code 证据 |
| `reflection` | 反思结果 | 用于 M_reflect 通道 |

---

### 2. `agent_pred.xlsx` - Agent 预测结果文件

Excel 文件，用于 EM 算法中的 **C 通道 (Agent Case-Level Score)**。

| 列名 | 类型 | 说明 |
|------|------|------|
| `case_name` | string | 任务唯一标识，如 `web_0_1` |
| `os_agent_score` | int | Agent 判定结果: 1=成功(DONE), 0=失败 |
| `evidence` | json string | 证据详情，格式见下 |

**evidence 字段格式**：

```jsonc
{
  "0": {
    "result": "Pass",                  // Pass 或 Fail
    "evidence": "任务完成的证据描述...",
    "case_desc": "任务描述/指令"        // 可选
  }
}
```

---

### 3. `code_evidence.json` - Code Evidence 数据

基于代码审查生成的测试用例及实现验证结果，用于 **E2_code** 证据通道。

```jsonc
[
  {
    "test_case": "导航验证：在页面滚动过程中，验证顶部导航栏是否保持固定位置",
    "related_requirement": "实现固定顶部导航栏",
    "requirement_id": "1",
    "code_review": {
      "is_implemented": true,           // 代码中是否实现该功能
      "evidence": "在 Header.tsx 中使用了 position: fixed 样式..."
    }
  },
  {
    "test_case": "响应式设计：验证技能标签云在不同设备和分辨率下的适配性",
    "related_requirement": "支持响应式布局",
    "requirement_id": "5",
    "code_review": {
      "is_implemented": false,
      "evidence": "代码中未发现媒体查询或响应式布局相关实现"
    }
  }
  // ... 更多测试用例 (通常 10-20 个)
]
```

**关键字段用途**：

| 字段 | 用途 | 说明 |
|------|------|------|
| `test_case` | 测试用例描述 | 从用户角度描述的 GUI 功能验证步骤 |
| `related_requirement` | 对应需求 | 该测试用例验证的编程需求 |
| `requirement_id` | 需求编号 | 用于关联和追踪 |
| `code_review.is_implemented` | **E2_code** 证据 | `true`=代码已实现, `false`=未实现 |
| `code_review.evidence` | 实现证据 | 代码片段或实现情况说明 |

---

## 🔧 使用方法

### 场景 1: AppEvalPilot Agent (有 info.txt 元素树)

AppEvalPilot 的 info.txt 包含完整的元素树信息，可以直接进行精确匹配。

```bash
cd gui_evidence/

# 使用 gui_evidence_generator.py
python gui_evidence_generator.py \
    --info-file /path/to/material/info.txt \
    --material-dir /path/to/material \
    --output gui_evidence.jsonl \
    --project-name "task_001" \
    --disable-mllm-fallback  # 禁用 MLLM 回退，仅用元素树匹配
```

**输入文件结构**：
```
material/
├── info.txt              # 包含元素树信息的日志文件
├── origin_1.jpg          # 第1步截图
├── origin_2.jpg          # 第2步截图
└── ...
```

**info.txt 格式要点**：
- 按 `#### iter:N` 分块
- 包含 `### Screenshot information ###` 元素树信息
- 包含 `### History operations ###` 操作历史

---

### 场景 2: UI-TARS / 其他 Agent (无元素树，使用 OCR)

对于 UI-TARS 等不输出元素树的 Agent，使用 `ui_tars_evidence_generator.py`，通过 OCR + 图标检测提取屏幕元素。

```bash
cd gui_evidence/

# 使用 ui_tars_evidence_generator.py
python ui_tars_evidence_generator.py \
    --data-dir /data/WebDevJudgeUnit_test \
    --output-dir /data/evidence_output \
    --output-filename gui_evidence.jsonl \
    --limit 10  # 可选：限制处理数量
```

**输入文件结构 (UI-TARS 格式)**：
```
WebDevJudgeUnit_test/
├── web_0/
│   ├── task_0/
│   │   ├── messages.json       # 对话和操作记录 (含 trajectory)
│   │   ├── metadata.json       # 任务元数据 (web_id, task_id, instruction)
│   │   └── screenshots/
│   │       ├── screenshot_001.png
│   │       ├── screenshot_002.png
│   │       └── ...
│   └── task_1/
│       └── ...
└── web_1/
    └── ...
```

**messages.json 格式要点**：
```jsonc
{
  "final_result": "DONE",           // 任务最终结果
  "trajectory": [                   // 操作轨迹
    {
      "role": "assistant",
      "content": "Thought: ... Action: click(450, 722)"
    },
    // ...
  ]
}
```

---

### 场景 3: 生成 Agent 预测文件

```bash
# 生成 agent_pred.xlsx
python generate_agent_pred_file.py
# 默认输入: /data/WebDevJudgeUnit_test
# 默认输出: /data/WebDevJudgeUnit_test/webdevjudge_pred.xlsx
```

---

### 场景 4: 生成 Code Evidence (测试用例 + 代码审查)

Code Evidence 生成器通过 LLM 分析项目需求和源代码，自动生成测试用例并审查代码实现情况。

**前置条件**：
```bash
# 1. 确保已配置 API 密钥
export LLM_API_BASE="https://your-api-endpoint.com"
export LLM_API_KEY="your-api-key-here"
```

**使用方法**：
```bash
cd code_evidence/

# 2. 准备输入文件
# - requirements.json: 项目需求列表
# - 项目源代码目录

# 3. 修改 test_case_generation.py 中的路径配置
# REQUIREMENTS_FILE = "./your_requirements.json"
# PROJECT_ROOT_PATH = "/path/to/your/projects/"
# OUTPUT_DIR = "/path/to/output/"

# 4. 运行生成器
python test_case_generation.py
```

**输入文件结构**：

`requirements.json` 格式：
```jsonc
[
  {
    "project_name": "my_web_project",
    "requirements": [
      "1. 实现响应式导航栏",
      "2. 添加用户登录功能",
      "3. 支持深色模式切换"
    ]
  }
  // ... 更多项目
]
```

项目目录结构：
```
projects/
├── my_web_project/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   └── ...
│   │   └── App.tsx
│   └── package.json
└── another_project/
    └── ...
```

---

## 🔗 与 EM 自适应学习模块的关联


**配置文件位置**: `src/config.py`

```python
# GUI Evidence 文件路径
WEBDEVJUDGE_GUI_EVIDENCE_JSONL = WEBDEVJUDGE_TRAJ_DIR / "baseline_agent_ui_tars" / \
    "20251212_213137_gui_evidence.jsonl"

# Agent 预测文件路径  
WEBDEVJUDGE_PRED_FILE = WEBDEVJUDGE_TRAJ_DIR / "baseline_agent_ui_tars" / \
    "webdevjudge_pred.xlsx"
```

---

## 📊 Evidence 字段与 EM 模型对应关系

| Evidence 字段 | EM 模型通道 | 数据来源 | 说明 |
|--------------|------------|----------|------|
| `coordinate_match` | **E1_gui** | gui_evidence | GUI 点击坐标是否命中目标元素 |
| `code_review.is_implemented` | **E2_code** | code_evidence | 代码中是否实现了对应功能 |
| `reflection` | **M_reflect** | gui_evidence | 反思 Mask，控制是否使用反思证据 |
| `os_agent_score` | **C (Agent Score)** | agent_pred | Case-Level Agent 判定分数 |

EM 模型 (`em_evidencedh_refine.py`) 使用这些证据进行根因分析：
- **EnvFail (δ=0)**: 环境失败，非 Agent 责任
- **AgentFail (δ=1)**: Agent 失败，需要改进


---


## 完整流程

```bash
# 0. 配置 API (必须)
export LLM_API_BASE="https://your-api-endpoint.com"
export LLM_API_KEY="your-api-key-here"

# 1. 生成 GUI Evidence
cd gui_evidence
python ui_tars_evidence_generator.py \
    --data-dir /data/WebDevJudgeUnit_test \
    --output-filename gui_evidence.jsonl

# 2. 生成 Code Evidence
cd ../code_evidence
python test_case_generation.py

# 3. 生成 Agent Pred 文件
cd ..
python generate_agent_pred_file.py

# 4. 运行 EM 根因分析
cd ../..
python run_rootcase.py --data-path /path/to/merged_data.xlsx
```

---

## 🔍 扩展支持新 Agent

要支持新的 Agent，需要：

1. **确定数据格式**: 
   - 是否有元素树信息 → 可直接使用/修改 `gui_evidence_generator.py`
   - 仅有截图 → 使用 `ui_tars_evidence_generator.py` + OCR

2. **解析操作日志**:
   - 提取点击坐标 `click_coords`
   - 提取操作描述 `operation_desc`
   - 提取动作内容 `action_content`

3. **输出格式兼容**:
   - 确保输出的 `gui_evidence.jsonl` 包含 `element_distance_sorting` 和 `coordinate_match` 字段
   - 确保 `agent_pred.xlsx` 包含 `case_name`, `os_agent_score`, `evidence` 字段
