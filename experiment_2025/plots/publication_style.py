"""Shared typography and legend placement for the manuscript figures."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.text import Text

FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]


def set_style():
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
    })


def format_figure(fig, *, legend_columns=2, legend_size=8):
    """Remove the overall title; retain panel identifiers and data labels.

    One shared, white legend sits at the upper left of the first panel.
    Polar panels use the upper-left figure margin to keep all sectors visible.
    """
    set_style()
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    first = fig.axes[0]
    handles, labels = first.get_legend_handles_labels()
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if handles:
        kwargs = dict(loc="upper left", ncol=legend_columns, fontsize=legend_size,
                      frameon=True, facecolor="white", edgecolor="black",
                      framealpha=1, fancybox=False, borderpad=0.4,
                      handlelength=2, columnspacing=1)
        if first.name == "polar":
            fig.legend(handles, labels, bbox_to_anchor=(0.02, 0.995), **kwargs)
        else:
            first.legend(handles, labels, **kwargs)
    for text in fig.findobj(Text):
        text.set_fontfamily(FONT_FAMILY)
