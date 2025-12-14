#!/usr/bin/env python3
"""
UI-TARS 数据处理脚本

功能：
1. 遍历所有 web_*/task_* 目录
2. 对每个 screenshot 使用 screenshot2info.py 提取 OCR/图标信息
3. 从 messages.json 提取对应的操作信息 (Thought + Action)
4. 生成 evidence 数据

输出格式：
生成单个 gui_evidence.jsonl 文件，每行是一个完整的 task 数据，格式与
webdevjudge 的 gui_evidence.jsonl 兼容，包含：
- project_name: 项目名称
- project_folder: 项目文件夹路径
- status: 处理状态
- results: 所有迭代的结果列表 (包含 click_coords, element_distance_sorting 等)
- summary: 摘要统计
- coordinate_analysis: 坐标分析统计
"""

from screenshot2info import ScreenshotInfoExtractor
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re

# 添加必要的路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/root/tangjingyu/AppEval/AppEvalPilot')


@dataclass
class IterResult:
    """单次迭代的结果数据"""
    iter_num: int                               # 迭代编号
    reflection: Optional[str]                   # 反思内容
    click_coords: Optional[List[int]]           # 点击坐标 [x, y]
    operation_desc: Optional[str]               # 操作描述
    action_content: Optional[str]               # 执行的动作内容
    reflection_thought: Optional[str]           # 反思思考
    coordinate_match: Optional[int]             # 坐标匹配 (1 或 None)
    coordinate_analysis: Optional[Dict]         # 坐标分析结果
    element_distance_sorting: Optional[List[Dict]]  # 元素距离排序


@dataclass
class TaskEvidence:
    """整个任务的 evidence 数据"""
    project_name: str                           # 项目名称
    project_folder: str                         # 项目文件夹路径
    status: str                                 # 状态 (success/error)
    error: Optional[str]                        # 错误信息
    results: List[Dict]                         # 迭代结果列表
    processed_at: str                           # 处理时间
    summary: Dict                               # 摘要统计
    coordinate_analysis: Optional[Dict]         # 坐标分析统计


class RawDataProcessor:
    """ UI-TARS Data Processor"""

    def __init__(
        self,
        data_dir: str,
        output_dir: str = None,
        use_ocr: bool = True,
        use_icon_detect: bool = True,
        quad_split_ocr: bool = False,
        location_info: str = "center"
    ):
        """
        初始化处理器

        Args:
            data_dir: 数据根目录
            output_dir: 输出目录，默认为 data_dir 下的 evidence_output
            use_ocr: 是否使用 OCR
            use_icon_detect: 是否使用图标检测
            quad_split_ocr: 是否使用四象限 OCR
            location_info: 坐标格式 (center 或 bbox)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(
            output_dir) if output_dir else self.data_dir / "evidence_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 screenshot2info 提取器
        self.extractor = ScreenshotInfoExtractor(
            use_ocr=use_ocr,
            use_icon_detect=use_icon_detect,
            use_xml=False,  # Web 截图不使用 XML 元素树
            quad_split_ocr=quad_split_ocr,
            location_info=location_info,
            platform="Windows"  # 不影响，因为不使用 XML
        )

        # 配置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def find_all_tasks(self) -> List[Path]:
        """查找所有 task 目录"""
        tasks = []

        # 遍历所有 web_* 目录
        for web_dir in sorted(self.data_dir.glob("web_*")):
            if not web_dir.is_dir():
                continue

            # 遍历所有 task_* 目录
            for task_dir in sorted(web_dir.glob("task_*")):
                if not task_dir.is_dir():
                    continue

                # 检查必要文件是否存在
                messages_file = task_dir / "messages.json"
                screenshots_dir = task_dir / "screenshots"

                if messages_file.exists() and screenshots_dir.exists():
                    tasks.append(task_dir)

        self.logger.info(f"找到 {len(tasks)} 个有效的 task 目录")
        return tasks

    def parse_action(self, action_text: str) -> Tuple[str, Optional[str], Optional[List[int]]]:
        """
        解析 Action 文本，提取动作类型、目标和坐标

        Args:
            action_text: Action 文本，如 "click(point='<point>450 722</point>')", 
                         "type(content='xxx')", "scroll(point='<point>x y</point>', direction='down')"

        Returns:
            (action_type, action_target, click_coords) 元组
        """
        action_text = action_text.strip()
        click_coords = None

        # 匹配 click(point='<point>x y</point>')
        click_match = re.match(
            r"click\s*\(\s*point\s*=\s*['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]", action_text)
        if click_match:
            x, y = float(click_match.group(1)), float(click_match.group(2))
            click_coords = [int(x), int(y)]
            return "click", f"({int(x)}, {int(y)})", click_coords

        # 匹配 left_double(point='<point>x y</point>')
        double_click_match = re.match(
            r"left_double\s*\(\s*point\s*=\s*['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]", action_text)
        if double_click_match:
            x, y = float(double_click_match.group(1)), float(
                double_click_match.group(2))
            click_coords = [int(x), int(y)]
            return "left_double", f"({int(x)}, {int(y)})", click_coords

        # 匹配 right_single(point='<point>x y</point>')
        right_click_match = re.match(
            r"right_single\s*\(\s*point\s*=\s*['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]", action_text)
        if right_click_match:
            x, y = float(right_click_match.group(1)), float(
                right_click_match.group(2))
            click_coords = [int(x), int(y)]
            return "right_single", f"({int(x)}, {int(y)})", click_coords

        # 匹配 drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
        drag_match = re.match(
            r"drag\s*\(\s*start_point\s*=\s*['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]"
            r"\s*,\s*end_point\s*=\s*['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]",
            action_text
        )
        if drag_match:
            x1, y1 = float(drag_match.group(1)), float(drag_match.group(2))
            x2, y2 = float(drag_match.group(3)), float(drag_match.group(4))
            click_coords = [int(x1), int(y1)]  # 使用起始点作为坐标
            return "drag", f"({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})", click_coords

        # 匹配 type(content='xxx')
        type_match = re.match(
            r"type\s*\(\s*content\s*=\s*['\"](.+?)['\"]\s*\)", action_text, re.DOTALL)
        if type_match:
            return "type", type_match.group(1), None

        # 匹配 scroll(point='<point>x y</point>', direction='down')
        scroll_match = re.match(
            r"scroll\s*\(\s*point\s*=\s*['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]"
            r"\s*,\s*direction\s*=\s*['\"](\w+)['\"]",
            action_text
        )
        if scroll_match:
            x, y = float(scroll_match.group(1)), float(scroll_match.group(2))
            direction = scroll_match.group(3)
            click_coords = [int(x), int(y)]
            return "scroll", f"({int(x)}, {int(y)}); {direction}", click_coords

        # 匹配 hotkey(key='xxx')
        hotkey_match = re.match(
            r"hotkey\s*\(\s*key\s*=\s*['\"](.+?)['\"]\s*\)", action_text)
        if hotkey_match:
            return "hotkey", hotkey_match.group(1), None

        # 匹配 wait()
        if action_text.lower().startswith('wait'):
            return "wait", None, None

        # 匹配 finished(content='xxx')
        finished_match = re.match(
            r"finished\s*\(\s*content\s*=\s*['\"](.+?)['\"]\s*\)", action_text, re.DOTALL)
        if finished_match:
            return "finished", finished_match.group(1), None

        # 其他情况
        action_type = action_text.split('(')[0] if '(' in action_text else action_text.split()[
            0] if action_text else "Unknown"
        return action_type, action_text, None

    def extract_thought_and_action(self, content: str) -> Tuple[str, str]:
        """
        从 assistant 消息中提取 Thought 和 Action

        Args:
            content: assistant 消息内容

        Returns:
            (thought, action) 元组
        """
        thought = ""
        action = ""

        # 提取 Thought
        thought_match = re.search(
            r'Thought:\s*(.*?)(?=\nAction:|$)', content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(r'Action:\s*(.*?)$', content, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def process_screenshot(self, screenshot_path: str) -> Tuple[str, List[Dict]]:
        """
        处理单个截图，提取信息

        Args:
            screenshot_path: 截图路径

        Returns:
            (格式化的截图信息, 原始感知信息列表)
        """
        try:
            # 提取感知信息
            perception_infos = self.extractor.extract(screenshot_path)

            # 格式化输出
            from PIL import Image
            image = Image.open(screenshot_path)
            width, height = image.size

            formatted_info = self.extractor.format_output(
                perception_infos, width, height
            )

            return formatted_info, perception_infos

        except Exception as e:
            self.logger.error(f"处理截图失败 {screenshot_path}: {e}")
            return "", []

    def extract_click_coords_from_codes(self, codes: List[str]) -> List[Optional[List[int]]]:
        """
        从 codes 数组中提取所有点击坐标

        Args:
            codes: codes 数组，如 ["\npyautogui.click(864.0, 779.76, button='left')", 'FAILED']

        Returns:
            点击坐标列表，每个元素是 [x, y] 或 None
        """
        coords_list = []
        for code in codes:
            if not isinstance(code, str):
                continue
            # 匹配 pyautogui.click(x, y, button='left') 格式
            click_match = re.search(
                r"pyautogui\.click\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)", code)
            if click_match:
                x, y = float(click_match.group(1)), float(click_match.group(2))
                coords_list.append([int(x), int(y)])
        return coords_list

    def process_single_task(self, task_dir: Path) -> Optional[TaskEvidence]:
        """
        处理单个 task，生成任务级别的 evidence

        Args:
            task_dir: task 目录路径

        Returns:
            TaskEvidence 对象或 None
        """
        from datetime import datetime

        # 读取 messages.json
        messages_file = task_dir / "messages.json"
        try:
            with open(messages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"读取 {messages_file} 失败: {e}")
            return None

        trajectory = data.get('trajectory', [])
        codes = data.get('codes', [])
        final_result = data.get('final_result', '')

        # 获取所有截图文件 (WebDevJudgeUnit 使用 screenshot_001.png 格式)
        screenshots_dir = task_dir / "screenshots"
        screenshot_files = sorted(
            screenshots_dir.glob("screenshot_*.png"),
            key=lambda x: int(re.search(r'(\d+)', x.stem).group(1))
        )

        # 提取所有 assistant 消息
        assistant_messages = [
            item for item in trajectory
            if item.get('role') == 'assistant'
        ]

        # 从 codes 中提取实际执行的点击坐标
        codes_coords = self.extract_click_coords_from_codes(codes)

        # 确保截图和消息数量匹配 (截图数量通常比 assistant 消息多1，因为第一张是初始状态)
        # WebDevJudgeUnit: screenshot_001 是初始状态，每个 action 后有一张新截图
        num_steps = len(assistant_messages)

        if len(screenshot_files) < num_steps:
            self.logger.warning(
                f"{task_dir.name}: 截图数量({len(screenshot_files)}) < "
                f"消息数量({num_steps}), 使用 {len(screenshot_files)} 步"
            )
            num_steps = len(screenshot_files)

        # 构建 project_name，格式为 web_{web_num}_{task_num}
        # 从目录名中提取 web 和 task 编号
        web_dir_name = task_dir.parent.name  # 如 "web_0"
        task_dir_name = task_dir.name  # 如 "task_1"

        # 提取数字部分
        web_match = re.search(r'web_(\d+)', web_dir_name)
        task_match = re.search(r'task_(\d+)', task_dir_name)

        if web_match and task_match:
            web_num = web_match.group(1)
            task_num = task_match.group(1)
            project_name = f"web_{web_num}_{task_num}"
        else:
            # 回退到原来的逻辑
            rel_path = task_dir.relative_to(self.data_dir)
            project_name = str(rel_path).replace(os.sep, '_')

        results = []
        coordinate_cases = 0
        reflection_count = 0

        # 处理每一步
        for step_idx in range(num_steps):
            # 截图索引：第一个 assistant 消息对应 screenshot_001 (action 执行前的状态)
            screenshot_path = screenshot_files[step_idx]
            assistant_content = assistant_messages[step_idx].get('content', '')

            # 提取 Thought 和 Action
            thought, action = self.extract_thought_and_action(
                assistant_content)

            # 解析 Action 类型并获取坐标
            action_type, action_target, click_coords = self.parse_action(
                action)

            # 如果从 action 中没有提取到坐标，尝试从 codes 中获取
            if click_coords is None and step_idx < len(codes_coords):
                click_coords = codes_coords[step_idx]

            # 处理截图获取感知信息
            screenshot_info, perception_infos = self.process_screenshot(
                str(screenshot_path))

            # 构建元素距离排序列表
            element_distance_sorting = None
            if perception_infos and click_coords:
                element_distance_sorting = self._build_element_distance_sorting(
                    perception_infos, click_coords
                )
                coordinate_cases += 1

            # 处理 action_content，最后一步需要转换为 Tell 格式
            final_action_content = action if action else None
            if step_idx == num_steps - 1:
                # 最后一步转换为 Tell 格式
                final_action_content = self._format_tell_action(
                    final_result, thought, action
                )

            # 创建迭代结果
            iter_result = {
                "iter_num": step_idx + 1,
                "reflection": None,  # WebDevJudgeUnit 数据中一般没有 reflection
                "click_coords": click_coords,
                "operation_desc": thought if thought else None,
                "action_type": action_type,
                "action_target": action_target,
                "action_content": final_action_content,
                "reflection_thought": thought if thought else None,
                "coordinate_match": 1 if click_coords and element_distance_sorting else None,
                "coordinate_analysis": {
                    "accuracy": 1,
                    "method": "element_tree",
                    "matched_element_id": element_distance_sorting[0]["id"] if element_distance_sorting else None
                } if click_coords and element_distance_sorting else None,
                "element_distance_sorting": element_distance_sorting,
                "final_result": final_result if step_idx == num_steps - 1 else None
            }

            results.append(iter_result)

        # 构建摘要
        summary = {
            "total_iters": len(results),
            "reflection_found": reflection_count,
            "reflection_error_cases": 0,
            "coordinate_analyzed": coordinate_cases
        }

        # 构建坐标分析统计
        coord_analysis = {
            "total_coordinate_cases": coordinate_cases,
            "successfully_analyzed": coordinate_cases,
            "coordinate_analysis_dir": str(task_dir / "coordinate_analysis"),
            "coordinate_files_created": coordinate_cases,
            "element_tree_cases": coordinate_cases * 2,
            "element_tree_matched": coordinate_cases,
            "mllm_cases": 0
        } if coordinate_cases > 0 else None

        # 创建 TaskEvidence
        task_evidence = TaskEvidence(
            project_name=project_name,
            project_folder=str(task_dir),
            status="success",
            error=None,
            results=results,
            processed_at=datetime.now().isoformat(),
            summary=summary,
            coordinate_analysis=coord_analysis
        )

        return task_evidence

    def _format_tell_action(
        self,
        final_result: str,
        thought: str,
        action: str
    ) -> str:
        """
        将最后一步的结果转换为 Tell 格式

        Args:
            final_result: 最终结果状态 (DONE, FAILED, INITIAL_ERROR 等)
            thought: 思考内容
            action: 原始动作内容

        Returns:
            Tell 格式的 action_content
        """
        # 根据 final_result 判断 Pass/Fail
        if final_result and final_result.upper() == 'DONE':
            result = 'Pass'
        else:
            result = 'Fail'

        # 从 action 中提取 finished 的内容作为 evidence 的一部分
        evidence = thought if thought else ""

        # 如果 action 是 finished(content='...') 格式，提取其中的内容
        finished_match = re.match(
            r"finished\s*\(\s*content\s*=\s*['\"](.+?)['\"]\s*\)", action if action else "", re.DOTALL
        )
        if finished_match:
            finished_content = finished_match.group(1)
            if evidence:
                evidence = f"{evidence} Final result: {finished_content}"
            else:
                evidence = finished_content

        # 如果没有 evidence，使用 final_result 作为 evidence
        if not evidence:
            evidence = f"Task completed with status: {final_result}"

        # 构建 Tell 格式
        tell_data = {
            "0": {
                "result": result,
                "evidence": evidence
            }
        }
        return f"Tell ({json.dumps(tell_data)})"

    def _build_element_distance_sorting(
        self,
        perception_infos: List[Dict],
        click_coords: List[int]
    ) -> List[Dict]:
        """
        构建元素距离排序列表

        Args:
            perception_infos: 感知信息列表
            click_coords: 点击坐标 [x, y]

        Returns:
            按距离排序的元素列表
        """
        import math

        elements = []
        click_x, click_y = click_coords

        for idx, info in enumerate(perception_infos):
            # 获取元素边界框
            bbox = info.get('coordinates', info.get('bbox', [0, 0, 0, 0]))
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                continue

            # 计算中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 计算距离
            distance = math.sqrt((click_x - center_x) **
                                 2 + (click_y - center_y) ** 2)

            # 计算面积
            area = abs((x2 - x1) * (y2 - y1))

            # 判断点击点是否在元素内部
            is_inside = x1 <= click_x <= x2 and y1 <= click_y <= y2

            element = {
                "id": f"element_{idx}",
                "ui_name": info.get('text', ''),
                "control_type": info.get('type', 'Unknown'),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "distance": distance,
                "area": int(area),
                "is_inside": is_inside
            }
            elements.append(element)

        # 按距离排序
        elements.sort(key=lambda x: x['distance'])

        return elements

    def save_evidence(self, task_evidence: TaskEvidence, output_file: Path):
        """
        将单个任务的 evidence 追加到输出文件

        Args:
            task_evidence: TaskEvidence 对象
            output_file: 输出文件路径
        """
        # 转换为 dict
        evidence_dict = asdict(task_evidence)

        # 追加到 JSONL 文件
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(evidence_dict, f, ensure_ascii=False)
            f.write('\n')

        self.logger.debug(f"已保存 {task_evidence.project_name} 到 {output_file}")

    def process_all(self, limit: int = None, output_filename: str = None) -> Dict:
        """
        处理所有 task

        Args:
            limit: 限制处理的 task 数量 (用于测试)
            output_filename: 输出文件名 (默认: gui_evidence.jsonl)

        Returns:
            处理统计信息
        """
        from datetime import datetime

        tasks = self.find_all_tasks()

        if limit:
            tasks = tasks[:limit]
            self.logger.info(f"限制处理前 {limit} 个 task")

        # 生成输出文件路径
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_gui_evidence.jsonl"

        output_file = self.output_dir / output_filename

        # 如果文件存在，先清空
        if output_file.exists():
            output_file.unlink()

        stats = {
            'total_tasks': len(tasks),
            'processed_tasks': 0,
            'failed_tasks': 0,
            'total_iters': 0,
            'total_coordinate_cases': 0
        }

        for i, task_dir in enumerate(tasks):
            self.logger.info(
                f"处理 [{i+1}/{len(tasks)}]: {task_dir.relative_to(self.data_dir)}")

            try:
                task_evidence = self.process_single_task(task_dir)

                if task_evidence:
                    self.save_evidence(task_evidence, output_file)
                    stats['processed_tasks'] += 1
                    stats['total_iters'] += len(task_evidence.results)
                    if task_evidence.coordinate_analysis:
                        stats['total_coordinate_cases'] += task_evidence.coordinate_analysis.get(
                            'total_coordinate_cases', 0)
                else:
                    stats['failed_tasks'] += 1

            except Exception as e:
                self.logger.error(f"处理失败 {task_dir}: {e}")
                import traceback
                traceback.print_exc()
                stats['failed_tasks'] += 1

        stats['output_file'] = str(output_file)
        return stats

    def print_summary(self, stats: Dict):
        """打印处理统计"""
        print("\n" + "=" * 50)
        print("处理统计")
        print("=" * 50)
        print(f"总任务数: {stats['total_tasks']}")
        print(f"成功处理: {stats['processed_tasks']}")
        print(f"处理失败: {stats['failed_tasks']}")
        print(f"总迭代数: {stats['total_iters']}")
        print(f"坐标分析数: {stats['total_coordinate_cases']}")
        print(f"输出文件: {stats.get('output_file', self.output_dir)}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='WebDevJudgeUnit 数据处理脚本 - 生成 Evidence'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/WebDevJudgeUnit_test',
        help='数据根目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录 (默认: data_dir/evidence_output)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的 task 数量 (用于测试)'
    )
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='禁用 OCR'
    )
    parser.add_argument(
        '--no-icon',
        action='store_true',
        help='禁用图标检测'
    )
    parser.add_argument(
        '--quad-split',
        action='store_true',
        help='使用四象限 OCR'
    )
    parser.add_argument(
        '--location',
        type=str,
        choices=['center', 'bbox'],
        default='center',
        help='坐标格式'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='详细日志'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='输出文件名 (默认: <timestamp>_gui_evidence.jsonl)'
    )

    args = parser.parse_args()

    # 配置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在 {args.data_dir}")
        sys.exit(1)

    # 创建处理器
    processor = RawDataProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_ocr=not args.no_ocr,
        use_icon_detect=not args.no_icon,
        quad_split_ocr=args.quad_split,
        location_info=args.location
    )

    # 处理所有任务
    stats = processor.process_all(
        limit=args.limit,
        output_filename=args.output_filename
    )

    # 打印统计
    processor.print_summary(stats)


if __name__ == "__main__":
    main()
