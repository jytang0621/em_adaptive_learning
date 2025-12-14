from appeval.tools.icon_detect import IconDetector
from appeval.tools.ocr import OCRTool
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


class ScreenshotInfoExtractor:
    """截图信息提取器

    从截图中提取以下信息:
    1. OCR 文本识别
    2. 图标检测
    3. 元素树（仅 Windows/Android 平台）
    """

    def __init__(
        self,
        use_ocr: bool = True,
        use_icon_detect: bool = True,
        use_xml: bool = False,
        quad_split_ocr: bool = False,
        location_info: str = "center",
        platform: str = "Windows"
    ):
        """初始化提取器

        Args:
            use_ocr: 是否使用 OCR 识别
            use_icon_detect: 是否使用图标检测
            use_xml: 是否使用元素树（需要连接设备）
            quad_split_ocr: 是否分割 OCR（4象限处理）
            location_info: 坐标格式，'center' 或 'bbox'
            platform: 平台类型，'Windows' 或 'Android'
        """
        self.use_ocr = use_ocr
        self.use_icon_detect = use_icon_detect
        self.use_xml = use_xml
        self.quad_split_ocr = quad_split_ocr
        self.location_info = location_info
        self.platform = platform

        # 初始化工具
        self.ocr_tool = None
        self.icon_detector = None
        self.controller = None

        if self.use_ocr:
            try:
                self.ocr_tool = OCRTool()
            except Exception as e:
                print(f"[警告] OCR 工具初始化失败: {e}")
                self.use_ocr = False

        if self.use_icon_detect:
            try:
                self.icon_detector = IconDetector()
            except Exception as e:
                print(f"[警告] 图标检测器初始化失败: {e}")
                self.use_icon_detect = False

        if self.use_xml:
            try:
                if platform == "Windows":
                    from appeval.tools.device_controller import PCController
                    self.controller = PCController()
                elif platform == "Android":
                    from appeval.tools.device_controller import AndroidController
                    self.controller = AndroidController()
            except Exception as e:
                print(f"[警告] 设备控制器初始化失败: {e}")
                self.use_xml = False

    def extract(self, screenshot_path: str) -> List[Dict]:
        """从截图中提取信息

        Args:
            screenshot_path: 截图文件路径

        Returns:
            List[Dict]: 感知信息列表，每个元素包含:
                - coordinates: 坐标 [x, y] 或 [x1, y1, x2, y2]
                - text: 内容描述
                - source: 来源 ('ocr', 'icon', 'xml')
        """
        perception_infos = []

        # 获取图片尺寸
        try:
            image = Image.open(screenshot_path)
            width, height = image.size
        except Exception as e:
            print(f"[错误] 无法打开图片: {e}")
            return []

        # 1. OCR 文本识别
        if self.use_ocr and self.ocr_tool:
            try:
                texts, text_coordinates = self.ocr_tool.ocr(
                    screenshot_path,
                    split=self.quad_split_ocr
                )
                for i, (text, coords) in enumerate(zip(texts, text_coordinates)):
                    if self.location_info == "center":
                        x1, y1, x2, y2 = coords
                        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        perception_infos.append({
                            "coordinates": center,
                            "text": f"text: {text}",
                            "source": "ocr"
                        })
                    else:
                        perception_infos.append({
                            "coordinates": coords,
                            "text": f"text: {text}",
                            "source": "ocr"
                        })
            except Exception as e:
                print(f"[警告] OCR 处理失败: {e}")

        # 2. 图标检测
        if self.use_icon_detect and self.icon_detector:
            try:
                icon_coordinates = self.icon_detector.detect(screenshot_path)
                for i, coords in enumerate(icon_coordinates):
                    if self.location_info == "center":
                        x1, y1, x2, y2 = coords
                        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        perception_infos.append({
                            "coordinates": center,
                            "text": "icon",
                            "source": "icon"
                        })
                    else:
                        perception_infos.append({
                            "coordinates": coords,
                            "text": "icon",
                            "source": "icon"
                        })
            except Exception as e:
                print(f"[警告] 图标检测失败: {e}")

        # 3. 元素树（需要连接设备）
        if self.use_xml and self.controller:
            try:
                xml_results = self.controller.get_screen_xml(
                    self.location_info)
                for info in xml_results:
                    info["source"] = "xml"
                    perception_infos.append(info)
            except Exception as e:
                print(f"[警告] 元素树获取失败: {e}")

        return perception_infos

    def format_output(
        self,
        perception_infos: List[Dict],
        width: int,
        height: int,
        instruction: str = ""
    ) -> str:
        """格式化输出为 Screenshot information 格式

        Args:
            perception_infos: 感知信息列表
            width: 图片宽度
            height: 图片高度
            instruction: 用户指令（可选）

        Returns:
            str: 格式化的 Screenshot information
        """
        # 坐标格式说明
        location_format = {
            "center": "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom;",
            "bbox": "The format of the coordinates is [x1, y1, x2, y2], x is the pixel from left to right and y is the pixel from top to bottom. (x1, y1) is the coordinates of the upper-left corner, (x2, y2) is the coordinates of the bottom-right corner;",
        }[self.location_info]

        # 内容格式说明
        content_format = """the content can be one of three types:
1. text from OCR
2. icon description or 'icon'
3. element information from device tree, which contains element attributes like type, text content, identifiers, accessibility descriptions and position information"""

        # 构建元素信息
        clickable_info_lines = []
        for info in perception_infos:
            if info["text"] and info["text"] != "icon: None" and info["coordinates"] != (0, 0):
                clickable_info_lines.append(
                    f"{info['coordinates']}; {info['text']}")

        clickable_info = "\n".join(clickable_info_lines)

        # 构建完整输出
        output = f"""### Screenshot information ###

In order to help you better perceive the content in this screenshot, we extract some information of the current screenshot.
This information pertains ONLY to the latest screenshot.
This information consists of two parts: coordinates; content. 
{location_format}
{content_format}
The information is as follow:
{clickable_info}
Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."""

        return output

    def extract_and_format(self, screenshot_path: str, instruction: str = "") -> str:
        """提取并格式化截图信息

        Args:
            screenshot_path: 截图文件路径
            instruction: 用户指令（可选）

        Returns:
            str: 格式化的 Screenshot information
        """
        # 获取图片尺寸
        try:
            image = Image.open(screenshot_path)
            width, height = image.size
        except Exception as e:
            return f"[错误] 无法打开图片: {e}"

        # 提取信息
        perception_infos = self.extract(screenshot_path)

        # 格式化输出
        return self.format_output(perception_infos, width, height, instruction)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从截图中提取 Screenshot information（OCR、图标检测、元素树）"
    )
    parser.add_argument(
        "screenshot",
        type=str,
        help="截图文件路径"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="禁用 OCR 识别"
    )
    parser.add_argument(
        "--no-icon",
        action="store_true",
        help="禁用图标检测"
    )
    parser.add_argument(
        "--use-xml",
        action="store_true",
        help="启用元素树提取（需要连接设备）"
    )
    parser.add_argument(
        "--quad-split",
        action="store_true",
        help="使用四象限分割 OCR（适用于大图）"
    )
    parser.add_argument(
        "--location",
        type=str,
        choices=["center", "bbox"],
        default="center",
        help="坐标格式: center (中心点) 或 bbox (边界框)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["Windows", "Android"],
        default="Windows",
        help="平台类型（用于元素树提取）"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="输出文件路径（默认输出到标准输出）"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出原始 JSON 格式（不格式化）"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.screenshot).exists():
        print(f"[错误] 文件不存在: {args.screenshot}")
        sys.exit(1)

    # 初始化提取器
    extractor = ScreenshotInfoExtractor(
        use_ocr=not args.no_ocr,
        use_icon_detect=not args.no_icon,
        use_xml=args.use_xml,
        quad_split_ocr=args.quad_split,
        location_info=args.location,
        platform=args.platform
    )

    # 提取信息
    if args.json:
        import json
        perception_infos = extractor.extract(args.screenshot)
        output = json.dumps(perception_infos, ensure_ascii=False, indent=2)
    else:
        output = extractor.extract_and_format(args.screenshot)

    # 输出结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[成功] 结果已保存到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
