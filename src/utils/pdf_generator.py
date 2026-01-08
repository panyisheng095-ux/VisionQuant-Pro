from fpdf import FPDF
import datetime
import os
import platform


class QuantPDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.font_path = self._get_chinese_font()
        # 注册字体 (关键步骤)
        if self.font_path:
            self.add_font('ChineseFont', '', self.font_path, uni=True)
            self.has_font = True
        else:
            self.has_font = False

    def _get_chinese_font(self):
        """自动寻找系统中的中文字体"""
        system = platform.system()
        # 常见的中文字体路径
        paths = [
            "/System/Library/Fonts/PingFang.ttc",  # Mac
            "/System/Library/Fonts/STHeiti Light.ttc",  # Mac
            "/Library/Fonts/Arial Unicode.ttf",  # Mac Office
            "C:\\Windows\\Fonts\\simhei.ttf",  # Windows
            "C:\\Windows\\Fonts\\msyh.ttf",  # Windows
            "./SimHei.ttf"  # 项目根目录自定义
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def header(self):
        if self.has_font:
            self.set_font('ChineseFont', '', 16)
        else:
            self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'VisionQuant Pro 智能投研报告', 0, 1, 'C')
        self.ln(5)


def generate_report_pdf(symbol, report_data, plot_path, output_path):
    pdf = QuantPDFReport()
    pdf.add_page()

    # 设置正文字体
    if pdf.has_font:
        pdf.set_font('ChineseFont', '', 10)
    else:
        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 10, "Warning: Chinese font not found. Characters may be missing.", 0, 1)

    # 1. 基础信息
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, f"标的: {symbol} | 生成日期: {datetime.datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'L', True)

    # 2. 决策结论
    pdf.ln(5)
    pdf.set_font(size=14)
    pdf.cell(0, 10, f"最终决策: {report_data.action}", 0, 1)
    pdf.set_font(size=10)
    pdf.cell(0, 8, f"信心指数: {report_data.confidence}/100  |  风险评估: {report_data.risk_level}", 0, 1)

    # 3. 插入图片
    if plot_path and os.path.exists(plot_path):
        pdf.ln(5)
        # 调整图片大小以适应页面 (A4 宽约 210mm)
        pdf.image(plot_path, x=10, w=190)

    # 4. 分析理由
    pdf.ln(10)
    pdf.set_font(size=12)
    pdf.cell(0, 10, "深度研判逻辑:", 0, 1)
    pdf.set_font(size=10)

    # 写入长文本
    text = report_data.reasoning
    # 如果没有中文字体，临时转码防止报错
    if not pdf.has_font:
        text = text.encode('latin-1', 'replace').decode('latin-1')

    pdf.multi_cell(0, 6, text)

    pdf.output(output_path)
    print(f"PDF Generated: {output_path}")