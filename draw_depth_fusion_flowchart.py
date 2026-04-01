from pathlib import Path

import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "docs" / "figures"
PNG_PATH = FIGURE_DIR / "3dgs_depth_fusion_flowchart.png"
PDF_PATH = FIGURE_DIR / "3dgs_depth_fusion_flowchart.pdf"
SVG_PATH = FIGURE_DIR / "3dgs_depth_fusion_flowchart.svg"
HTML_PATH = FIGURE_DIR / "3dgs_depth_fusion_flowchart.html"
FIG_WIDTH = 1800
FIG_HEIGHT = 1120


def _html(text: str) -> str:
    return text.replace("\n", "<br>")


def add_box(fig, x, y, w, h, text, fill, line, font_size=18, weight="normal", font_color="#1b1f24"):
    if weight == "bold":
        text = f"<b>{_html(text)}</b>"
    else:
        text = _html(text)
    fig.add_shape(
        type="rect",
        x0=x,
        y0=y,
        x1=x + w,
        y1=y + h,
        xref="paper",
        yref="paper",
        line=dict(color=line, width=2),
        fillcolor=fill,
        layer="below",
    )
    fig.add_annotation(
        x=x + w / 2,
        y=y + h / 2,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        text=text,
        font=dict(size=font_size, color=font_color, family="DejaVu Sans"),
    )


def add_panel(fig, x, y, w, h, title, fill, line, title_color):
    fig.add_shape(
        type="rect",
        x0=x,
        y0=y,
        x1=x + w,
        y1=y + h,
        xref="paper",
        yref="paper",
        line=dict(color=line, width=2.5),
        fillcolor=fill,
        layer="below",
    )
    fig.add_annotation(
        x=x + 0.015,
        y=y + h - 0.018,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        text=f"<b>{title}</b>",
        font=dict(size=24, color=title_color, family="DejaVu Sans"),
    )


def add_arrow(fig, start, end, color, text=None, text_pos=0.5, text_offset=(0.0, 0.0), width=2.2):
    ax = (start[0] - end[0]) * FIG_WIDTH
    ay = (end[1] - start[1]) * FIG_HEIGHT
    fig.add_annotation(
        x=end[0],
        y=end[1],
        ax=ax,
        ay=ay,
        xref="paper",
        yref="paper",
        axref="pixel",
        ayref="pixel",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=width,
        arrowcolor=color,
        text="",
    )
    if text:
        tx = start[0] + (end[0] - start[0]) * text_pos + text_offset[0]
        ty = start[1] + (end[1] - start[1]) * text_pos + text_offset[1]
        fig.add_annotation(
            x=tx,
            y=ty,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=text,
            font=dict(size=16, color=color, family="DejaVu Sans"),
        )


def build_figure():
    fig = go.Figure()
    fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_layout(
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#f7f8fb",
        plot_bgcolor="#f7f8fb",
    )

    fig.add_annotation(
        x=0.04,
        y=0.965,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        text="<b>3DGS Rendering Path + Monocular Depth Supervision Path</b>",
        font=dict(size=32, color="#162033", family="DejaVu Sans"),
    )
    fig.add_annotation(
        x=0.04,
        y=0.925,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        align="left",
        text=(
            "Improved FlashAvatar training architecture: 3D Gaussian Splatting shares one geometry state, "
            "while SIDL depth loss regularizes the same Gaussian centers through camera projection."
        ),
        font=dict(size=17, color="#5d6778", family="DejaVu Sans"),
    )

    add_panel(fig, 0.56, 0.52, 0.38, 0.33, "3DGS Rendering Path", "#edf5ff", "#9bc3ea", "#2d6aa6")
    add_panel(fig, 0.56, 0.13, 0.38, 0.30, "Monocular Depth Supervision Path (SIDL)", "#eef8ea", "#a8cfa0", "#467b43")

    add_box(
        fig,
        0.05,
        0.61,
        0.15,
        0.20,
        "Inputs\n\nRGB frames\nalpha / head mask /\nmouth mask\nFLAME + camera\ntracking\noptional mono depth",
        "#fff1df",
        "#de9e49",
        font_size=18,
        weight="bold",
    )
    add_box(
        fig,
        0.25,
        0.69,
        0.19,
        0.11,
        "Scene_mica loader\nframe-aligned Camera\nGT image + masks\nmono_depth + valid_mask",
        "#e8ecff",
        "#7786d7",
        font_size=17,
        weight="bold",
    )
    add_box(
        fig,
        0.25,
        0.52,
        0.19,
        0.11,
        "Deform_Model decode\nshape + expr + jaw + eyes + eyelids\n-> verts_final, rot_delta, scale_coef",
        "#efe5ff",
        "#9f74cc",
        font_size=17,
        weight="bold",
    )
    add_box(
        fig,
        0.25,
        0.33,
        0.19,
        0.12,
        "GaussianModel update\ncreate / update xyz, rotation, scale\nshared geometry + appearance state",
        "#e5f7f3",
        "#48a795",
        font_size=17,
        weight="bold",
    )
    add_box(
        fig,
        0.25,
        0.10,
        0.19,
        0.13,
        "Loss Fusion + Backprop\nL_total = L_rgb + lambda_depth * L_depth\nupdate Gaussian parameters\nand Deform_Model weights",
        "#ffe8ed",
        "#d36d85",
        font_size=17,
        weight="bold",
    )
    add_box(
        fig,
        0.60,
        0.66,
        0.14,
        0.10,
        "render(viewpoint_cam,\ngaussians, pipe,\nbackground)",
        "#ffffff",
        "#6ea6d8",
        font_size=16,
    )
    add_box(
        fig,
        0.79,
        0.66,
        0.11,
        0.10,
        "RGB render\nI_hat",
        "#ffffff",
        "#6ea6d8",
        font_size=18,
        weight="bold",
    )
    add_box(
        fig,
        0.66,
        0.54,
        0.20,
        0.09,
        "RGB reconstruction losses\nHuber(I_hat, I_gt)\n+ 40 x mouth Huber\n+ LPIPS after 15k iters",
        "#ffffff",
        "#6ea6d8",
        font_size=15,
    )
    add_box(
        fig,
        0.59,
        0.25,
        0.15,
        0.10,
        "gaussians.get_xyz\n-> homogeneous points\n-> camera projection",
        "#ffffff",
        "#7fb072",
        font_size=16,
    )
    add_box(
        fig,
        0.78,
        0.25,
        0.13,
        0.10,
        "sample mono depth\nwith bilinear\ngrid_sample",
        "#ffffff",
        "#7fb072",
        font_size=16,
    )
    add_box(
        fig,
        0.65,
        0.15,
        0.22,
        0.07,
        "valid region = eroded head/neck mask - mouth interior",
        "#ffffff",
        "#7fb072",
        font_size=15,
        weight="bold",
    )
    add_box(
        fig,
        0.63,
        0.04,
        0.27,
        0.08,
        "Scale-Invariant Depth Loss\nL_depth = 1 - corr(z_render, z_mono)\nPearson correlation on valid samples only",
        "#ffffff",
        "#7fb072",
        font_size=15,
    )
    add_box(
        fig,
        0.05,
        0.14,
        0.15,
        0.12,
        "Key idea\nNo extra mesh fitting branch.\nDepth supervision regularizes\nthe same Gaussian centers\nalready used by 3DGS.",
        "#ffffff",
        "#c9cdd4",
        font_size=15,
    )

    add_arrow(fig, (0.20, 0.71), (0.25, 0.745), "#7b7f87")
    add_arrow(fig, (0.345, 0.69), (0.345, 0.63), "#7b7f87")
    add_arrow(fig, (0.345, 0.52), (0.345, 0.45), "#7b7f87")
    add_arrow(fig, (0.345, 0.33), (0.345, 0.23), "#7b7f87")

    add_arrow(
        fig,
        (0.44, 0.39),
        (0.60, 0.71),
        "#2d6aa6",
        text="shared Gaussian state",
        text_pos=0.48,
        text_offset=(0.02, 0.02),
    )
    add_arrow(fig, (0.74, 0.71), (0.79, 0.71), "#2d6aa6")
    add_arrow(fig, (0.845, 0.66), (0.76, 0.63), "#2d6aa6")
    add_arrow(
        fig,
        (0.66, 0.54),
        (0.44, 0.165),
        "#2d6aa6",
        text="L_rgb",
        text_pos=0.40,
        text_offset=(0.02, 0.02),
    )

    add_arrow(
        fig,
        (0.44, 0.39),
        (0.59, 0.30),
        "#467b43",
        text="xyz / pose-aware geometry",
        text_pos=0.45,
        text_offset=(0.01, -0.02),
    )
    add_arrow(fig, (0.74, 0.30), (0.78, 0.30), "#467b43")
    add_arrow(fig, (0.845, 0.25), (0.76, 0.22), "#467b43")
    add_arrow(fig, (0.76, 0.15), (0.76, 0.12), "#467b43")
    add_arrow(
        fig,
        (0.63, 0.08),
        (0.44, 0.16),
        "#467b43",
        text="L_depth",
        text_pos=0.42,
        text_offset=(0.0, -0.025),
    )
    add_arrow(
        fig,
        (0.44, 0.745),
        (0.78, 0.25),
        "#85aa7e",
        text="mono_depth + valid_mask",
        text_pos=0.45,
        text_offset=(0.02, 0.01),
        width=1.8,
    )

    fig.add_annotation(
        x=0.04,
        y=0.025,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        align="left",
        text=(
            "Fusion logic: the RGB branch enforces view-consistent appearance via 3DGS rendering, while the SIDL branch "
            "projects the same Gaussian centers into the camera plane and aligns rendered depth with monocular depth "
            "only on conservative valid pixels."
        ),
        font=dict(size=16, color="#586172", family="DejaVu Sans"),
    )

    return fig


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.write_html(HTML_PATH, include_plotlyjs="cdn")
    print(f"[Flowchart] Saved: {HTML_PATH}")

    for path in (PNG_PATH, PDF_PATH, SVG_PATH):
        try:
            fig.write_image(path)
            print(f"[Flowchart] Saved: {path}")
        except Exception as exc:
            print(f"[Flowchart][WARN] Failed to save {path}: {exc}")


if __name__ == "__main__":
    main()
