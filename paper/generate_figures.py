#!/usr/bin/env python3
"""Generate academic-style figures for the MRIxFields MMFM paper.

Figures:
  1. figure1_overview.png      — challenge task and CFM concept
  2. figure2_mmfm_v1.png       — vectorized MMFM baseline
  3. figure3_mmfm_unet_v2.png  — spatial UNet MMFM v2
  4. figure4_training_inference.png — random crop and patch-based inference

Style: academic scientific, minimal, Arial/Helvetica, panel labels (a, b, c).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# --------------------------------------------------------------------------- #
# Academic style setup
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 300,
})

COLORS = {
    "source": "#4C78A8",
    "target": "#E45756",
    "vae": "#F58518",
    "latent": "#54A24B",
    "flow": "#79706E",
    "mlp": "#B279A2",
    "unet": "#72B7B2",
    "attention": "#Eeca3b",
    "brain": "#D8D8D8",
    "crop": "#72B7B2",
    "patch": "#4C78A8",
    "bg": "#FFFFFF",
    "text": "#333333",
}


def add_panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(
        x, y, f"({label})", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="right",
        color=COLORS["text"]
    )


def draw_box(ax, x, y, w, h, color, text, text_color="white", fontsize=8, alpha=1.0):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=color, edgecolor="#333333", linewidth=0.7, alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", wrap=True)


def draw_arrow(ax, start, end, color="#333333", style="->", lw=1.2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"))


def draw_curly_brace(ax, x1, y1, x2, y2, text, color="#333333", fontsize=8):
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.plot([x1, x1, (x1 + mid_x) / 2, mid_x, (mid_x + x2) / 2, x2, x2],
            [y1, mid_y, (y1 + mid_y) / 2, mid_y, (mid_y + y2) / 2, mid_y, y2],
            color=color, lw=0.8)
    ax.text(mid_x, mid_y + 0.02, text, ha="center", va="bottom",
            fontsize=fontsize, color=color)


def draw_brain_schematic(ax, x, y, w, h, color, label=""):
    """Draw a simple elliptical brain silhouette."""
    theta = np.linspace(0, 2 * np.pi, 200)
    # Rough brain-like shape: two ellipses merged
    ax.fill(x + w * np.cos(theta), y + 0.6 * h * np.sin(theta),
            color=color, edgecolor="#555555", linewidth=0.6)
    ax.text(x, y - h * 0.55, label, ha="center", va="top", fontsize=8,
            color=COLORS["text"])


def figure1_overview():
    """Figure 1: MRIxFields challenge overview and CFM concept."""
    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.25, left=0.04, right=0.96, top=0.88, bottom=0.12)

    # (a) Task
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "MRIxFields 2026 task", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COLORS["text"])

    fields = ["0.1T", "1.5T", "3T", "5T", "7T"]
    mods = ["T1W", "T2W", "T2FLAIR"]
    start_y = 0.78
    for i, mod in enumerate(mods):
        for j, field in enumerate(fields):
            x = 0.10 + j * 0.18
            y = start_y - i * 0.23
            c = COLORS["source"] if (i, j) != (1, 4) else COLORS["target"]
            is_target = (i, j) == (1, 4)
            draw_brain_schematic(ax, x, y, 0.08, 0.13, c,
                                 label="" if not is_target else "")
            if is_target:
                ax.text(x + 0.10, y - 0.02, f"{mod} @ {field}", ha="left", va="center",
                        fontsize=8, color=COLORS["target"], fontweight="bold")
    draw_arrow(ax, (0.46, 0.36), (0.74, 0.36), color=COLORS["target"], lw=2)
    ax.text(0.60, 0.32, "any-to-any\ntranslation", ha="center", va="top",
            fontsize=8, color=COLORS["target"], fontweight="bold")
    ax.text(0.5, 0.06, "Translate any source domain into any target domain",
            ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    # (b) What
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "b", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "What: flow matching in VAE latent space",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    draw_box(ax, 0.25, 0.72, 0.35, 0.18, COLORS["source"], "source\nvolume $x_{\\mathrm{src}}$")
    draw_arrow(ax, (0.45, 0.60), (0.45, 0.48))
    draw_box(ax, 0.25, 0.36, 0.35, 0.18, COLORS["vae"], "MedVAE\n$z_{\\mathrm{src}}$")
    draw_arrow(ax, (0.45, 0.24), (0.45, 0.12), color=COLORS["flow"], lw=2)
    ax.text(0.55, 0.18, "$v_\\theta(z_t,t,y)$", fontsize=8, color=COLORS["flow"], fontweight="bold")
    draw_box(ax, 0.25, -0.06, 0.35, 0.18, COLORS["target"], "target latent\n$z_{\\mathrm{tgt}}$")

    # (c) Why + How
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "c", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "How: OT-CFM objective", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COLORS["text"])

    # Equation
    eq = (r"$\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t,(z_0,z_1) \sim \pi}"
          r"\left[\|v_\theta(z_t,t,y) - u_t\|_2^2\right]$")
    ax.text(0.5, 0.78, eq, ha="center", va="center", fontsize=10, color=COLORS["text"])

    ax.text(0.5, 0.58, r"$z_t = (1-t)z_0 + t z_1$", ha="center", va="center",
            fontsize=9, color=COLORS["text"])
    ax.text(0.5, 0.48, r"$u_t = z_1 - z_0$", ha="center", va="center",
            fontsize=9, color=COLORS["text"])

    # Simple trajectory sketch
    t = np.linspace(0, 1, 100)
    ax.plot(t, 0.2 + 0.15 * t, color=COLORS["flow"], lw=2)
    ax.scatter([0], [0.2], color=COLORS["source"], s=60, zorder=5)
    ax.scatter([1], [0.35], color=COLORS["target"], s=60, zorder=5)
    ax.text(0.5, 0.32, "learned velocity field", ha="center", va="bottom",
            fontsize=8, color=COLORS["flow"])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.1, 0.45)
    ax.set_xlabel("time $t$", fontsize=8)
    ax.set_ylabel("latent state", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["src", "tgt"])
    ax.set_yticks([])

    fig.savefig("paper/figure1_overview.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved paper/figure1_overview.png")


def figure2_mmfm_v1():
    """Figure 2: MMFM v1 vectorized."""
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.35,
                          left=0.04, right=0.96, top=0.90, bottom=0.08)

    # (a) What
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "What: vectorized latent flow",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    draw_box(ax, 0.25, 0.74, 0.35, 0.16, COLORS["source"], "$z \\in \\mathbb{R}^{1\\times32\\times32\\times20}$")
    draw_arrow(ax, (0.45, 0.63), (0.45, 0.53))
    ax.text(0.62, 0.58, "flatten", fontsize=8, color=COLORS["text"], fontweight="bold")
    draw_box(ax, 0.25, 0.40, 0.35, 0.16, COLORS["latent"], "$z_{\\mathrm{vec}} \\in \\mathbb{R}^{20480}$")
    draw_arrow(ax, (0.45, 0.29), (0.45, 0.19), color=COLORS["mlp"], lw=2)
    ax.text(0.62, 0.24, "vector\nvelocity", fontsize=8, color=COLORS["mlp"], fontweight="bold")
    draw_box(ax, 0.25, 0.06, 0.35, 0.16, COLORS["target"], "$\\hat{z}_{\\mathrm{tgt,vec}} \\in \\mathbb{R}^{20480}$")

    # (b) Why
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "b", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "Why: simplicity and speed",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    bullets = [
        "• MedVAE kept frozen",
        "• No 3D convolutions",
        "• Compact MLP backbone",
        "• Fast training (~5 h for 30 k iters)",
        "• Strong baseline reference",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.08, 0.78 - i * 0.12, b, ha="left", va="top",
                fontsize=9, color=COLORS["text"])

    # (c) How - architecture
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "c", x=-0.03, y=1.05)
    ax.text(0.5, 0.95, "How: VectorMMFM architecture",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    # Input blocks
    draw_box(ax, 0.10, 0.58, 0.16, 0.18, COLORS["latent"], "$z_{t,\\mathrm{vec}}$", fontsize=8)
    draw_box(ax, 0.10, 0.30, 0.16, 0.18, COLORS["source"], "$z_{\\mathrm{src,vec}}$", fontsize=8)
    draw_box(ax, 0.10, 0.02, 0.16, 0.18, COLORS["flow"], "$\\gamma(t)$\n$e(y)$", fontsize=8)

    # Concat arrow
    ax.text(0.30, 0.48, "concat", ha="center", va="center", fontsize=8,
            color=COLORS["text"], fontweight="bold")
    draw_arrow(ax, (0.27, 0.58), (0.34, 0.48))
    draw_arrow(ax, (0.27, 0.30), (0.34, 0.48))
    draw_arrow(ax, (0.27, 0.02), (0.34, 0.48))

    # MLP blocks
    x0 = 0.38
    for i in range(4):
        draw_box(ax, x0 + i * 0.13, 0.48, 0.11, 0.22, COLORS["mlp"], f"ResBlock\n{i+1}", fontsize=7)
        if i < 3:
            draw_arrow(ax, (x0 + i * 0.13 + 0.055, 0.48), (x0 + (i+1) * 0.13 - 0.055, 0.48))

    # Output
    draw_arrow(ax, (x0 + 3 * 0.13 + 0.055, 0.48), (0.92, 0.48))
    draw_box(ax, 0.94, 0.48, 0.12, 0.22, COLORS["target"], "$v_t$", fontsize=9)

    fig.savefig("paper/figure2_mmfm_v1.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved paper/figure2_mmfm_v1.png")


def figure3_mmfm_unet_v2():
    """Figure 3: MMFM-UNet v2."""
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.35,
                          left=0.04, right=0.96, top=0.90, bottom=0.08)

    # (a) What
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "What: spatial latent flow",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    draw_box(ax, 0.25, 0.74, 0.35, 0.16, COLORS["source"], "source latent\n$z_{\\mathrm{src}}$")
    draw_arrow(ax, (0.45, 0.63), (0.45, 0.53))
    draw_box(ax, 0.25, 0.40, 0.35, 0.16, COLORS["unet"], "UNet velocity field\n$v_\\theta(z_t,t,y)$")
    draw_arrow(ax, (0.45, 0.29), (0.45, 0.19), color=COLORS["target"], lw=2)
    draw_box(ax, 0.25, 0.06, 0.35, 0.16, COLORS["target"], "target latent\n$z_{\\mathrm{tgt}}$")

    # (b) Why
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "b", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "Why: better spatial modeling",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    bullets = [
        "• Preserves 3D latent structure",
        "• Convolutions capture local anatomy",
        "• Factorized attention for global context",
        "• Random crops for full-brain diversity",
        "• Designed for full-resolution inference",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.08, 0.78 - i * 0.12, b, ha="left", va="top",
                fontsize=9, color=COLORS["text"])

    # (c) How - UNet + factorized attention
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "c", x=-0.03, y=1.05)
    ax.text(0.5, 0.95, "How: UNet v2 with factorized bottleneck attention",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    # UNet encoder (no overlap)
    enc_x = [0.10, 0.25, 0.40]
    enc_w = [0.13, 0.12, 0.10]
    enc_h = [0.50, 0.38, 0.26]
    for i, (x, w, h) in enumerate(zip(enc_x, enc_w, enc_h)):
        draw_box(ax, x, 0.55, w, h, COLORS["unet"], f"Down\n×{2**i}", fontsize=7)
        if i < 2:
            draw_arrow(ax, (x + w/2 + 0.01, 0.55), (enc_x[i+1] - enc_w[i+1]/2 - 0.01, 0.55))

    # Bottleneck attention
    draw_box(ax, 0.55, 0.55, 0.12, 0.28, COLORS["attention"],
             "Factorized\nAttn 3D", fontsize=7, text_color=COLORS["text"])
    draw_arrow(ax, (0.40 + 0.05, 0.55), (0.55 - 0.06, 0.55))

    # Decoder (no overlap)
    dec_x = [0.72, 0.84, 0.96]
    dec_w = [0.10, 0.12, 0.13]
    dec_h = [0.26, 0.38, 0.50]
    for i, (x, w, h) in enumerate(zip(dec_x, dec_w, dec_h)):
        draw_box(ax, x, 0.55, w, h, COLORS["unet"], f"Up\n×{2**(2-i)}", fontsize=7)
        if i > 0:
            draw_arrow(ax, (dec_x[i-1] + dec_w[i-1]/2 + 0.01, 0.55), (x - w/2 - 0.01, 0.55))
    draw_arrow(ax, (0.55 + 0.06, 0.55), (0.72 - 0.05, 0.55))

    # Skip connections (dashed arcs above)
    for i, (ex, ew, dx, dw) in enumerate(zip(enc_x, enc_w, dec_x[::-1], dec_w[::-1])):
        eh = enc_h[i]
        dh = dec_h[2 - i]
        y_top = 0.88
        ax.plot([ex, ex - 0.03 + i * 0.01, dx + dw + 0.03 - i * 0.01, dx + dw],
                [0.55 + eh/2 + 0.02, y_top, y_top, 0.55 + dh/2 + 0.02],
                color="#888888", lw=0.8, linestyle="--")

    # Factorized attention detail
    ax.text(0.50, 0.18, "Axial attention: H → W → D", ha="center", va="center",
            fontsize=8, color=COLORS["text"], fontweight="bold")
    ax.text(0.50, 0.10, "Replaces O(N⁶) full 3D attention with 3×O(N²) operations",
            ha="center", va="center", fontsize=8, color=COLORS["text"])

    fig.savefig("paper/figure3_mmfm_unet_v2.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved paper/figure3_mmfm_unet_v2.png")


def figure4_training_inference():
    """Figure 4: random crop and patch-based inference."""
    fig = plt.figure(figsize=(10, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.25, left=0.04, right=0.96, top=0.85, bottom=0.12)

    # (a) Center crop vs random crop
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "Why random crops?",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    # Big brain
    draw_brain_schematic(ax, 0.5, 0.55, 0.50, 0.55, COLORS["brain"])
    # Center crop
    rect = Rectangle((0.35, 0.45), 0.30, 0.20, linewidth=2,
                     edgecolor=COLORS["source"], facecolor="none", linestyle="--")
    ax.add_patch(rect)
    ax.text(0.50, 0.43, "center crop only", ha="center", va="top",
            fontsize=8, color=COLORS["source"], fontweight="bold")
    # Random crop
    rect2 = Rectangle((0.22, 0.35), 0.30, 0.20, linewidth=2,
                      edgecolor=COLORS["crop"], facecolor="none", linestyle="--")
    ax.add_patch(rect2)
    ax.text(0.37, 0.33, "random crop", ha="center", va="top",
            fontsize=8, color=COLORS["crop"], fontweight="bold")

    ax.text(0.5, 0.08, "80% random / 20% center\nimproves full-brain generalization",
            ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    # (b) Patch-based inference pipeline
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "b", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "How: full-resolution inference",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    draw_box(ax, 0.50, 0.78, 0.50, 0.14, COLORS["source"], "0.5 mm full brain")
    draw_arrow(ax, (0.50, 0.70), (0.50, 0.62))
    draw_box(ax, 0.50, 0.54, 0.50, 0.14, COLORS["vae"], "resample to 1 mm")
    draw_arrow(ax, (0.50, 0.46), (0.50, 0.38))
    draw_box(ax, 0.50, 0.30, 0.50, 0.14, COLORS["patch"], "128×128×80 patches")
    draw_arrow(ax, (0.50, 0.22), (0.50, 0.14))
    draw_box(ax, 0.50, 0.06, 0.50, 0.14, COLORS["target"], "blend → resample to 0.5 mm", fontsize=8)

    # (c) Blending
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "c", x=-0.05, y=1.05)
    ax.text(0.5, 0.95, "Patch blending",
            ha="center", va="top", fontsize=11, fontweight="bold", color=COLORS["text"])

    # Hann window visualization
    x = np.linspace(-1, 1, 100)
    hann = 0.5 * (1 + np.cos(np.pi * x))
    ax.fill_between((x + 1) / 2, hann * 0.4 + 0.45, 0.45,
                    color=COLORS["patch"], alpha=0.4)
    ax.plot((x + 1) / 2, hann * 0.4 + 0.45, color=COLORS["patch"], lw=2)
    ax.text(0.5, 0.40, "Hann window weights", ha="center", va="top",
            fontsize=8, color=COLORS["patch"], fontweight="bold")

    # Overlapping patches sketch
    for i, dx in enumerate([-0.10, 0.00, 0.10]):
        rect = Rectangle((0.35 + dx, 0.62), 0.25, 0.18, linewidth=1.5,
                         edgecolor=COLORS["patch"], facecolor=COLORS["patch"], alpha=0.2 + i*0.1)
        ax.add_patch(rect)
    ax.text(0.5, 0.58, "50% overlap", ha="center", va="top",
            fontsize=8, color=COLORS["text"])

    ax.text(0.5, 0.12, "Weighted average of\noverlapping predictions",
            ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    fig.savefig("paper/figure4_training_inference.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved paper/figure4_training_inference.png")


def main():
    figure1_overview()
    figure2_mmfm_v1()
    figure3_mmfm_unet_v2()
    figure4_training_inference()


if __name__ == "__main__":
    main()
