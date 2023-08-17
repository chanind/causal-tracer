from typing import Optional
import os

from matplotlib import pyplot as plt

from .HiddenFlow import HiddenFlow, LayerKind


def plot_hidden_flow_heatmap(
    result: HiddenFlow,
    savepdf: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    show: bool = True,
    font_family: str = "",
    color: Optional[str] = None,
) -> plt.Figure:
    differences = result.scores
    low_score = result.low_score
    answer = result.answer
    kind = result.kind
    labels = list(result.input_tokens)
    for i in range(*result.subject_range):
        labels[i] = labels[i] + "*"

    color_map: dict[LayerKind, str] = {
        "hidden": "Purples",
        "mlp": "Greens",
        "attention": "Reds",
    }

    with plt.rc_context(rc={"font.family": font_family}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap=color or color_map[kind],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        kindname = kind
        ax.set_title(f"Impact of restoring {kindname} after corrupted input")
        ax.set_xlabel(f"center of interval of restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        if show:
            plt.show()
        return fig
