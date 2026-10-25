from __future__ import annotations

from pathlib import Path
import textwrap
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "src" / "figure" / "figure2_3_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\simhei.ttf")

W, H = 2600, 1500
SCALE = 2

COLORS = {
    "bg": "#FFFFFF",
    "ink": "#202124",
    "muted": "#5F6368",
    "line": "#9AA0A6",
    "era_fill": "#EAF2FB",
    "era_stroke": "#4F81BD",
    "gdas_fill": "#EAF6EF",
    "gdas_stroke": "#5B9A6B",
    "merge_fill": "#F5F5F5",
    "merge_stroke": "#6C757D",
    "eval_fill": "#FFF6E5",
    "eval_stroke": "#D39B2A",
    "accent": "#8A5A44",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size * SCALE)


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in text.split("\n"):
        if not part:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(part, width=width, break_long_words=False))
    return lines


def draw_round_rect(draw: ImageDraw.ImageDraw, xy, radius, fill, outline, width=3):
    xy = tuple(int(v * SCALE) for v in xy)
    draw.rounded_rectangle(xy, radius=radius * SCALE, fill=fill, outline=outline, width=width * SCALE)


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    box,
    title: str,
    body: str = "",
    title_color: str = COLORS["ink"],
    body_color: str = COLORS["muted"],
    title_size: int = 25,
    body_size: int = 21,
    wrap: int = 17,
):
    x, y, w, h = box
    title_font = font(title_size, bold=True)
    body_font = font(body_size, bold=False)
    lines = [title]
    body_lines = wrap_text(body, wrap) if body else []

    title_h = title_size * 1.25
    body_h = body_size * 1.28
    total_h = title_h + (8 if body_lines else 0) + len(body_lines) * body_h
    cy = y + (h - total_h) / 2

    def center_line(txt, yy, fnt, color):
        bbox = draw.textbbox((0, 0), txt, font=fnt)
        tw = (bbox[2] - bbox[0]) / SCALE
        draw.text(((x + (w - tw) / 2) * SCALE, yy * SCALE), txt, font=fnt, fill=color)

    center_line(title, cy, title_font, title_color)
    cy += title_h + (8 if body_lines else 0)
    for line in body_lines:
        center_line(line, cy, body_font, body_color)
        cy += body_h


def draw_node(draw, box, title, body, fill, stroke, wrap=17):
    x, y, w, h = box
    draw_round_rect(draw, (x, y, x + w, y + h), 28, fill, stroke, 3)
    draw_text_center(draw, box, title, body, wrap=wrap)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=COLORS["line"], width=4):
    sx, sy = start
    ex, ey = end
    draw.line((sx * SCALE, sy * SCALE, ex * SCALE, ey * SCALE), fill=color, width=width * SCALE)
    # Arrowhead.
    import math

    angle = math.atan2(ey - sy, ex - sx)
    head = 18
    for delta in (math.pi * 0.82, -math.pi * 0.82):
        x = ex + head * math.cos(angle + delta)
        y = ey + head * math.sin(angle + delta)
        draw.line((ex * SCALE, ey * SCALE, x * SCALE, y * SCALE), fill=color, width=width * SCALE)


def svg_text(x, y, text, size=22, weight="400", color=COLORS["ink"], anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Microsoft YaHei, SimHei, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>'
    )


def svg_wrapped_text(box, title, body, wrap=17):
    x, y, w, h = box
    body_lines = wrap_text(body, wrap) if body else []
    title_h = 31
    body_h = 27
    total_h = title_h + (8 if body_lines else 0) + len(body_lines) * body_h
    cy = y + (h - total_h) / 2 + 25
    out = [svg_text(x + w / 2, cy, title, size=25, weight="700")]
    cy += title_h + (8 if body_lines else 0)
    for line in body_lines:
        out.append(svg_text(x + w / 2, cy, line, size=21, color=COLORS["muted"]))
        cy += body_h
    return "\n".join(out)


def svg_node(box, title, body, fill, stroke, wrap=17):
    x, y, w, h = box
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" ry="28" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>\n'
        + svg_wrapped_text(box, title, body, wrap)
    )


def svg_arrow(start, end, color=COLORS["line"], width=4):
    sx, sy = start
    ex, ey = end
    return (
        f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{color}" '
        f'stroke-width="{width}" marker-end="url(#arrow)"/>'
    )


def build():
    img = Image.new("RGB", (W * SCALE, H * SCALE), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((120 * SCALE, 82 * SCALE), "数据预处理与验证评估流程", font=font(38, True), fill=COLORS["ink"])
    draw.text(
        (120 * SCALE, 132 * SCALE),
        "ERA5 滞后起报与 GDAS 实时起报在统一 Pangu 输入、推理和近海风场验证框架下进行比较",
        font=font(23),
        fill=COLORS["muted"],
    )

    # Lane labels
    draw_round_rect(draw, (115, 205, 1245, 235), 12, "#F8FBFF", COLORS["era_stroke"], 2)
    draw_round_rect(draw, (1355, 205, 2485, 235), 12, "#F8FCF9", COLORS["gdas_stroke"], 2)
    draw.text((565 * SCALE, 200 * SCALE), "ERA5_Lagged 分支", font=font(23, True), fill=COLORS["era_stroke"])
    draw.text((1775 * SCALE, 200 * SCALE), "GDAS_Realtime 分支", font=font(23, True), fill=COLORS["gdas_stroke"])

    era_nodes = [
        ((150, 285, 360, 128), "ERA5 再分析资料", "目标时刻前 120 h\n多时次 NetCDF", COLORS["era_fill"], COLORS["era_stroke"]),
        ((150, 480, 360, 142), "变量提取与坐标统一", "MSLP、U10、V10、T2M\nZ/Q/T/U/V 13 层\n0–359.75°经度，纬度降序", COLORS["era_fill"], COLORS["era_stroke"]),
        ((150, 690, 360, 128), "Pangu 输入张量", "input_surface.npy\ninput_upper.npy\n(4,721,1440)/(5,13,721,1440)", COLORS["era_fill"], COLORS["era_stroke"]),
        ((150, 890, 360, 128), "滞后滚动推演", "先积分 120 h 到目标时刻\n再生成 1–72 h 预报", COLORS["era_fill"], COLORS["era_stroke"]),
    ]
    gdas_nodes = [
        ((1990, 285, 360, 128), "GDAS 分析场", "目标时刻 GRIB2/FNL\n00 UTC 实时起报", COLORS["gdas_fill"], COLORS["gdas_stroke"]),
        ((1990, 480, 360, 142), "变量映射与单位转换", "prmsl/10u/10v/2t\ngh→位势 Z\n经纬度排序与层次对齐", COLORS["gdas_fill"], COLORS["gdas_stroke"]),
        ((1990, 690, 360, 128), "Pangu 输入张量", "input_surface.npy\ninput_upper.npy\n同一变量顺序与网格", COLORS["gdas_fill"], COLORS["gdas_stroke"]),
        ((1990, 890, 360, 128), "实时直接起报", "从目标时刻直接生成\n1–72 h 预报", COLORS["gdas_fill"], COLORS["gdas_stroke"]),
    ]
    for box, title, body, fill, stroke in era_nodes + gdas_nodes:
        draw_node(draw, box, title, body, fill, stroke, wrap=18)

    for nodes in (era_nodes, gdas_nodes):
        for a, b in zip(nodes[:-1], nodes[1:]):
            ax, ay, aw, ah = a[0]
            bx, by, bw, bh = b[0]
            arrow(draw, (ax + aw / 2, ay + ah), (bx + bw / 2, by), COLORS["line"], 4)

    merge_nodes = [
        ((890, 450, 480, 126), "Pangu ONNX 推理", "1/3/6/24 h 模型分步调用\n时间链缓存避免重复计算", COLORS["merge_fill"], COLORS["merge_stroke"]),
        ((890, 655, 480, 118), "输出解码", "npy → NetCDF\nsurface/upper 输出变量", COLORS["merge_fill"], COLORS["merge_stroke"]),
        ((890, 850, 480, 126), "区域裁剪与海洋掩膜", "中国海及邻近海域\n剔除陆地与缺测格点", COLORS["merge_fill"], COLORS["merge_stroke"]),
        ((890, 1045, 480, 126), "观测与参考场验证", "ERA5 对应时刻场\n浮标/ICOADS/最佳路径", COLORS["eval_fill"], COLORS["eval_stroke"]),
        ((890, 1240, 480, 126), "统计指标与论文图表", "Bias、MAE、RMSE\n风向角差、分级误差、台风个例", COLORS["eval_fill"], COLORS["eval_stroke"]),
    ]
    for box, title, body, fill, stroke in merge_nodes:
        draw_node(draw, box, title, body, fill, stroke, wrap=19)
    for a, b in zip(merge_nodes[:-1], merge_nodes[1:]):
        ax, ay, aw, ah = a[0]
        bx, by, bw, bh = b[0]
        arrow(draw, (ax + aw / 2, ay + ah), (bx + bw / 2, by), COLORS["line"], 4)

    # Branch merge arrows.
    arrow(draw, (510, 954), (890, 513), COLORS["era_stroke"], 5)
    arrow(draw, (1990, 954), (1370, 513), COLORS["gdas_stroke"], 5)

    png = OUT_DIR / "figure2_3_data_processing_flowchart.png"
    pdf = OUT_DIR / "figure2_3_data_processing_flowchart.pdf"
    img.save(png, dpi=(320, 320))
    img.save(pdf, "PDF", resolution=320)

    # SVG export with editable text.
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M0,0 L12,6 L0,12 z" fill="{COLORS["line"]}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
        svg_text(120, 120, "数据预处理与验证评估流程", size=38, weight="700", anchor="start"),
        svg_text(120, 165, "ERA5 滞后起报与 GDAS 实时起报在统一 Pangu 输入、推理和近海风场验证框架下进行比较", size=23, color=COLORS["muted"], anchor="start"),
        f'<rect x="115" y="205" width="1130" height="30" rx="12" fill="#F8FBFF" stroke="{COLORS["era_stroke"]}" stroke-width="2"/>',
        f'<rect x="1355" y="205" width="1130" height="30" rx="12" fill="#F8FCF9" stroke="{COLORS["gdas_stroke"]}" stroke-width="2"/>',
        svg_text(680, 227, "ERA5_Lagged 分支", size=23, weight="700", color=COLORS["era_stroke"]),
        svg_text(1920, 227, "GDAS_Realtime 分支", size=23, weight="700", color=COLORS["gdas_stroke"]),
    ]
    for box, title, body, fill, stroke in era_nodes + gdas_nodes + merge_nodes:
        svg.append(svg_node(box, title, body, fill, stroke, wrap=18 if box[2] < 400 else 19))
    for nodes in (era_nodes, gdas_nodes, merge_nodes):
        for a, b in zip(nodes[:-1], nodes[1:]):
            ax, ay, aw, ah = a[0]
            bx, by, bw, bh = b[0]
            svg.append(svg_arrow((ax + aw / 2, ay + ah), (bx + bw / 2, by)))
    svg.append(svg_arrow((510, 954), (890, 513), COLORS["era_stroke"], 5))
    svg.append(svg_arrow((1990, 954), (1370, 513), COLORS["gdas_stroke"], 5))
    svg.append("</svg>")
    svg_path = OUT_DIR / "figure2_3_data_processing_flowchart.svg"
    svg_path.write_text("\n".join(svg), encoding="utf-8")

    print(png)
    print(pdf)
    print(svg_path)


if __name__ == "__main__":
    build()
