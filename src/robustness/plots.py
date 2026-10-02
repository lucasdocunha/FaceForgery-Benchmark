"""Small plots derived only from the same certified evaluation scores."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .provenance import write_csv
from .statistics import checked_predictions


def write_forensic_plots(frame, metrics, output, *, title, unit):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frame = checked_predictions(frame)
    output = Path(output)
    files = []
    figure, axis = plt.subplots(figsize=(4.5, 4))
    if frame.label.nunique() == 2:
        fpr, tpr, thresholds = roc_curve(frame.label, frame.p_fake, drop_intermediate=False)
        coordinates = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": [float(t) if np.isfinite(t) else None for t in thresholds]})
        name = f"{unit}_roc.csv"
        write_csv(output / name, coordinates)
        files.append(name)
        axis.plot(fpr, tpr, label=f"AUC {metrics['auc']:.3f}; EER {metrics['eer']:.3f}")
        axis.legend(loc="lower right", fontsize=8)
    else:
        axis.text(0.5, 0.5, "ROC undefined: one class", ha="center")
    axis.plot([0, 1], [0, 1], linestyle=":", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1))
    name = f"{unit}_roc.png"
    figure.tight_layout()
    figure.savefig(output / name, dpi=120)
    plt.close(figure)
    files.append(name)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    bins = np.linspace(0, 1, 21)
    for label, name in ((0, "Real"), (1, "Fake")):
        scores = frame.loc[frame.label.eq(label), "p_fake"]
        if len(scores):
            axis.hist(scores, bins=bins, alpha=0.55, label=f"{name} (n={len(scores)})")
    axis.axvline(metrics["threshold"], color="black", linestyle="--", label="Frozen val threshold")
    axis.set(xlabel="p_fake", ylabel="Observations", title=title, xlim=(0, 1))
    axis.legend(fontsize=8)
    name = f"{unit}_scores.png"
    figure.tight_layout()
    figure.savefig(output / name, dpi=120)
    plt.close(figure)
    files.append(name)

    figure, axis = plt.subplots(figsize=(4.5, 4))
    matrix = np.array(metrics["confusion_matrix_normalized"], dtype=float)
    shown = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.3f}" if np.isfinite(value) else "undefined", ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], xlabel="Predicted", ylabel="True", title=title)
    figure.colorbar(shown, ax=axis)
    name = f"{unit}_confusion.png"
    figure.tight_layout()
    figure.savefig(output / name, dpi=120)
    plt.close(figure)
    files.append(name)
    return files
