from __future__ import annotations

from typing import List, Tuple, Optional


def try_plot_roc_curve(fpr: List[float], tpr: List[float], title: str = "ROC", save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plot.")
        return
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=title)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        try:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"Saved plot to {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save plot to {save_path}: {exc}")
    plt.show()


def try_plot_pr_curves(series: List[Tuple[str, List[float], List[float]]], title: str = "Precision-Recall", save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plot.")
        return
    plt.figure(figsize=(5, 4))
    for label, recalls, precisions in series:
        # Sort by recall for nicer plots
        pts = sorted(zip(recalls, precisions))
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(xs, ys, label=label, linewidth=2.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        try:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"Saved plot to {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save plot to {save_path}: {exc}")
    plt.show()



