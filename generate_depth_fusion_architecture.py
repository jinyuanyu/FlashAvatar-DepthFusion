import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


NODES = [
    {
        "id": "inputs",
        "label": "RGB / mask /\nFLAME track",
        "xy": (0.9, 14.2),
        "size": (4.2, 1.35),
        "fill": "#DCEEFF",
        "badge": "IN",
        "badge_fill": "#4C90E8",
    },
    {
        "id": "loader",
        "label": "Scene Loader",
        "xy": (0.9, 11.95),
        "size": (4.2, 1.35),
        "fill": "#EEF1F4",
        "badge": "SC",
        "badge_fill": "#7E8B96",
    },
    {
        "id": "deform",
        "label": "FLAME MLP",
        "xy": (0.9, 9.7),
        "size": (4.2, 1.35),
        "fill": "#CFE8FF",
        "badge": "NN",
        "badge_fill": "#3D78C5",
    },
    {
        "id": "gaussian",
        "label": "Gaussian State\nxyz / rot / scale / SH",
        "xy": (0.9, 7.25),
        "size": (4.2, 1.55),
        "fill": "#CFE8FF",
        "badge": "GS",
        "badge_fill": "#3D78C5",
    },
    {
        "id": "renderer",
        "label": "3DGS Render",
        "xy": (0.9, 4.95),
        "size": (4.2, 1.35),
        "fill": "#CFE8FF",
        "badge": "RD",
        "badge_fill": "#3D78C5",
    },
    {
        "id": "rgb_loss",
        "label": "RGB Loss\nHuber + LPIPS",
        "xy": (0.9, 2.65),
        "size": (4.2, 1.55),
        "fill": "#E5F4E8",
        "badge": "RL",
        "badge_fill": "#4DAA72",
    },
    {
        "id": "depth_input",
        "label": "Mono Depth\n+ valid mask",
        "xy": (7.1, 13.05),
        "size": (4.0, 1.35),
        "fill": "#FFE7C2",
        "badge": "MD",
        "badge_fill": "#E6A445",
    },
    {
        "id": "depth_proj",
        "label": "Depth Sample",
        "xy": (7.1, 10.6),
        "size": (4.0, 1.35),
        "fill": "#FFE7C2",
        "badge": "DP",
        "badge_fill": "#E6A445",
    },
    {
        "id": "depth_loss",
        "label": "SIDL Loss\ncorr(D_r, D_m)",
        "xy": (7.1, 8.15),
        "size": (4.0, 1.35),
        "fill": "#FFE7C2",
        "badge": "DL",
        "badge_fill": "#E6A445",
    },
    {
        "id": "total_loss",
        "label": "Total Loss\nLrgb + lambda_d Ld",
        "xy": (7.1, 5.05),
        "size": (4.0, 1.35),
        "fill": "#D9F5E5",
        "badge": "L",
        "badge_fill": "#4DAA72",
    },
    {
        "id": "opt",
        "label": "Optimize\nMLP + appearance",
        "xy": (7.1, 2.75),
        "size": (4.0, 1.55),
        "fill": "#D9F5E5",
        "badge": "OP",
        "badge_fill": "#4DAA72",
    },
]


MAIN_FLOW = [
    ("inputs", "loader", ""),
    ("loader", "deform", ""),
    ("deform", "gaussian", ""),
    ("gaussian", "renderer", ""),
    ("renderer", "rgb_loss", "RGB"),
]


DEPTH_FLOW = [
    ("gaussian", "depth_proj", "xyz"),
    ("depth_proj", "depth_loss", "D_r"),
    ("depth_input", "depth_loss", "D_m + mask"),
]


FUSION_FLOW = [
    ("rgb_loss", "total_loss", ""),
    ("depth_loss", "total_loss", "lambda_d"),
    ("total_loss", "opt", ""),
]


def _node_map():
    return {node["id"]: node for node in NODES}


def _left_center(node):
    x, y = node["xy"]
    _, h = node["size"]
    return x, y + h / 2.0


def _right_center(node):
    x, y = node["xy"]
    w, h = node["size"]
    return x + w, y + h / 2.0


def _top_center(node):
    x, y = node["xy"]
    w, h = node["size"]
    return x + w / 2.0, y + h


def _bottom_center(node):
    x, y = node["xy"]
    w, _ = node["size"]
    return x + w / 2.0, y


def _tikz_node_lines():
    fill_map = {
        "#DCEEFF": "inputFill",
        "#EEF1F4": "sharedFill",
        "#CFE8FF": "renderFill",
        "#E5F4E8": "rgbFill",
        "#FFE7C2": "depthFill",
        "#D9F5E5": "fusionFill",
    }
    lines = []
    for node in NODES:
        x, y = node["xy"]
        w, h = node["size"]
        label = node["label"].replace("\n", r"\\")
        fill_name = fill_map[node["fill"]]
        lines.append(
            rf"\node[box, fill={fill_name}, minimum width={w}cm, minimum height={h}cm] ({node['id']}) at ({x + w / 2.0:.2f}, {y + h / 2.0:.2f}) {{{label}}};"
        )
        lines.append(
            rf"\definecolor{{{node['id']}Badge}}{{HTML}}{{{node['badge_fill'].lstrip('#').upper()}}}"
        )
        lines.append(
            rf"\node[circle, draw=white, line width=0.6pt, fill={node['id']}Badge, text=white, font=\bfseries\tiny, minimum size=0.42cm] at ({x + 0.34:.2f}, {y + h - 0.24:.2f}) {{{node['badge']}}};"
        )
    return lines


def build_tikz_document():
    lines = [
        r"\documentclass[tikz,border=6pt]{standalone}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta,backgrounds}",
        r"\begin{document}",
        r"\begin{tikzpicture}[",
        r"  x=1cm, y=1cm,",
        r"  >=Latex,",
        r"  font=\sffamily,",
        r"  box/.style={rounded corners=3pt, draw=black!45, line width=0.9pt, align=center},",
        r"  flow/.style={-Latex, line width=1.0pt, draw=black!70},",
        r"  update/.style={-Latex, line width=1.0pt, draw=black!55, dashed},",
        r"]",
        r"\definecolor{inputFill}{HTML}{DCEEFF}",
        r"\definecolor{sharedFill}{HTML}{EEF1F4}",
        r"\definecolor{renderFill}{HTML}{CFE8FF}",
        r"\definecolor{rgbFill}{HTML}{E5F4E8}",
        r"\definecolor{depthFill}{HTML}{FFE7C2}",
        r"\definecolor{fusionFill}{HTML}{D9F5E5}",
        r"\definecolor{renderBand}{HTML}{EEF6FF}",
        r"\definecolor{depthBand}{HTML}{FFF5E7}",
        r"\node[anchor=west, font=\bfseries\Large] at (0.5, 17.45) {3DGS + Monocular Depth Fusion};",
        r"\node[anchor=west, font=\small, text=black!65] at (0.5, 16.95) {Vertical layout with a clearer top-to-bottom logic chain};",
        r"\node[circle, draw=white, line width=0.5pt, fill=inputFill!70!blue, text=white, font=\bfseries\tiny, minimum size=0.34cm] at (7.7, 17.18) {IN};",
        r"\node[anchor=west, font=\scriptsize, text=black!70] at (7.93, 17.18) {input};",
        r"\node[circle, draw=white, line width=0.5pt, fill=renderFill!75!blue, text=white, font=\bfseries\tiny, minimum size=0.34cm] at (9.0, 17.18) {NN};",
        r"\node[anchor=west, font=\scriptsize, text=black!70] at (9.23, 17.18) {learned};",
        r"\node[circle, draw=white, line width=0.5pt, fill=depthFill!85!orange, text=white, font=\bfseries\tiny, minimum size=0.34cm] at (10.5, 17.18) {D};",
        r"\node[anchor=west, font=\scriptsize, text=black!70] at (10.73, 17.18) {depth};",
        r"\node[circle, draw=white, line width=0.5pt, fill=fusionFill!80!green!60!black, text=white, font=\bfseries\tiny, minimum size=0.34cm] at (11.75, 17.18) {L};",
        r"\node[anchor=west, font=\scriptsize, text=black!70] at (11.98, 17.18) {loss};",
    ]

    lines.extend(_tikz_node_lines())
    lines.extend(
        [
            r"\begin{scope}[on background layer]",
            r"\fill[renderBand, rounded corners=6pt] (0.55,2.15) rectangle (5.45,16.4);",
            r"\fill[depthBand, rounded corners=6pt] (6.75,7.4) rectangle (11.45,15.4);",
            r"\fill[fusionFill!30, rounded corners=6pt] (6.75,2.35) rectangle (11.45,6.75);",
            r"\node[anchor=west, font=\bfseries\small, text=blue!55!black] at (0.8, 16.18) {Main 3DGS chain};",
            r"\node[anchor=west, font=\bfseries\small, text=orange!70!black] at (7.0, 15.1) {Depth branch};",
            r"\node[anchor=west, font=\bfseries\small, text=green!45!black] at (7.0, 6.45) {Fusion + optimization};",
            r"\end{scope}",
        ]
    )

    for src, dst, label in MAIN_FLOW:
        if label:
            lines.append(
                rf"\draw[flow] ({src}.south) -- node[midway, fill=white, inner sep=1.2pt, font=\scriptsize] {{{label}}} ({dst}.north);"
            )
        else:
            lines.append(rf"\draw[flow] ({src}.south) -- ({dst}.north);")

    lines.append(
        r"\draw[flow] (gaussian.east) -- node[midway, fill=white, inner sep=1.2pt, font=\scriptsize] {xyz} (depth_proj.west);"
    )
    lines.append(
        r"\draw[flow] (depth_input.south) -- node[midway, fill=white, inner sep=1.2pt, font=\scriptsize] {D_m + mask} (depth_proj.north);"
    )
    lines.append(
        r"\draw[flow] (depth_proj.south) -- node[midway, fill=white, inner sep=1.2pt, font=\scriptsize] {D_r} (depth_loss.north);"
    )
    lines.append(
        r"\draw[flow] (depth_loss.south) -- node[midway, fill=white, inner sep=1.2pt, font=\scriptsize] {lambda_d} (total_loss.north);"
    )
    lines.append(
        r"\draw[flow] (rgb_loss.east) |- node[pos=0.30, fill=white, inner sep=1.2pt, font=\scriptsize] {Lrgb} (total_loss.west);"
    )
    lines.append(r"\draw[flow] (total_loss.south) -- (opt.north);")

    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines) + "\n"


def _draw_box(ax, node):
    x, y = node["xy"]
    w, h = node["size"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#4A4A4A",
        facecolor=node["fill"],
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        node["label"],
        ha="center",
        va="center",
        fontsize=11.0,
        color="#202020",
        zorder=3,
    )
    badge = Circle(
        (x + 0.32, y + h - 0.22),
        radius=0.18,
        facecolor=node["badge_fill"],
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    ax.add_patch(badge)
    ax.text(
        x + 0.32,
        y + h - 0.22,
        node["badge"],
        ha="center",
        va="center",
        fontsize=7.5,
        color="white",
        fontweight="bold",
        zorder=5,
    )


def _draw_arrow(ax, start, end, label="", dashed=False, offset=(0.0, 0.16)):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.15,
        color="#4F4F4F",
        linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=0.0",
        zorder=1.5,
    )
    ax.add_patch(arrow)
    if label:
        lx = (start[0] + end[0]) / 2.0 + offset[0]
        ly = (start[1] + end[1]) / 2.0 + offset[1]
        ax.text(
            lx,
            ly,
            label,
            fontsize=8.6,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            zorder=4,
        )


def render_preview(preview_path: Path):
    nodes = _node_map()
    fig, ax = plt.subplots(figsize=(8.6, 11.5))
    ax.set_xlim(0.0, 12.4)
    ax.set_ylim(-1.0, 18.1)
    ax.axis("off")

    main_band = FancyBboxPatch(
        (0.55, 2.15),
        4.9,
        14.25,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.0,
        edgecolor="#8FC0FF",
        facecolor="#EEF6FF",
        zorder=0,
    )
    depth_band = FancyBboxPatch(
        (6.75, 7.4),
        4.7,
        8.0,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.0,
        edgecolor="#F2B766",
        facecolor="#FFF5E7",
        zorder=0,
    )
    fusion_band = FancyBboxPatch(
        (6.75, 2.35),
        4.7,
        4.4,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.0,
        edgecolor="#7DCEA0",
        facecolor="#EFF9F2",
        zorder=0,
    )
    ax.add_patch(main_band)
    ax.add_patch(depth_band)
    ax.add_patch(fusion_band)
    ax.text(0.8, 16.18, "Main 3DGS chain", fontsize=12.5, fontweight="bold", color="#6AA9FF", va="top")
    ax.text(7.0, 15.1, "Depth branch", fontsize=12.5, fontweight="bold", color="#E5A74E", va="top")
    ax.text(7.0, 6.45, "Fusion + optimization", fontsize=12.2, fontweight="bold", color="#67B884", va="top")

    for node in NODES:
        _draw_box(ax, node)

    for src, dst, label in MAIN_FLOW:
        _draw_arrow(
            ax,
            _bottom_center(nodes[src]),
            _top_center(nodes[dst]),
            label=label,
            offset=(0.35, 0.0),
        )

    _draw_arrow(
        ax,
        _right_center(nodes["gaussian"]),
        _left_center(nodes["depth_proj"]),
        label="xyz",
        offset=(0.0, 0.22),
    )
    _draw_arrow(
        ax,
        _bottom_center(nodes["depth_input"]),
        _top_center(nodes["depth_proj"]),
        label="D_m + mask",
        offset=(0.48, 0.0),
    )
    _draw_arrow(
        ax,
        _bottom_center(nodes["depth_proj"]),
        _top_center(nodes["depth_loss"]),
        label="D_r",
        offset=(0.28, 0.0),
    )
    _draw_arrow(
        ax,
        _bottom_center(nodes["depth_loss"]),
        _top_center(nodes["total_loss"]),
        label="lambda_d",
        offset=(0.35, 0.0),
    )
    _draw_arrow(
        ax,
        _right_center(nodes["rgb_loss"]),
        _left_center(nodes["total_loss"]),
        label="Lrgb",
        offset=(0.0, 0.18),
    )
    _draw_arrow(ax, _bottom_center(nodes["total_loss"]), _top_center(nodes["opt"]))

    ax.text(0.5, 17.65, "3DGS + Monocular Depth Fusion", fontsize=22, fontweight="bold", va="top")
    ax.text(
        0.5,
        17.1,
        "Vertical layout with a clearer top-to-bottom logic chain",
        fontsize=11.5,
        color="#5A5A5A",
        va="top",
    )
    legend = [
        (8.75, "IN", "#4C90E8", "input"),
        (9.7, "NN", "#3D78C5", "learned"),
        (10.7, "D", "#E6A445", "depth"),
        (11.55, "L", "#4DAA72", "loss"),
    ]
    for x, badge_text, badge_fill, label in legend:
        circ = Circle((x, 17.05), radius=0.11, facecolor=badge_fill, edgecolor="white", linewidth=0.8, zorder=5)
        ax.add_patch(circ)
        ax.text(x, 17.05, badge_text, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold", zorder=6)
        ax.text(x + 0.16, 17.05, label, ha="left", va="center", fontsize=8.3, color="#666666", zorder=6)

    fig.tight_layout()
    fig.savefig(preview_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def maybe_compile_tex(tex_path: Path):
    engine = shutil.which("pdflatex") or shutil.which("xelatex")
    if engine is None:
        print("[DepthFusionDiagram] No LaTeX engine found; skipped PDF compilation.")
        return
    subprocess.run(
        [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[DepthFusionDiagram] Tried compiling with {Path(engine).name}.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simplified TikZ + Python architecture diagram for 3DGS and monocular-depth fusion."
    )
    parser.add_argument(
        "--outdir",
        default="figures/depth_fusion_architecture",
        help="Output directory for generated assets.",
    )
    parser.add_argument(
        "--compile-tex",
        action="store_true",
        help="Compile the TikZ file if pdflatex/xelatex is available.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tex_path = outdir / "depth_fusion_architecture.tex"
    preview_path = outdir / "depth_fusion_architecture_preview.png"

    tex_path.write_text(build_tikz_document(), encoding="utf-8")
    render_preview(preview_path)

    print(f"[DepthFusionDiagram] Wrote TikZ source: {tex_path}")
    print(f"[DepthFusionDiagram] Wrote preview PNG: {preview_path}")

    if args.compile_tex:
        maybe_compile_tex(tex_path)


if __name__ == "__main__":
    main()
