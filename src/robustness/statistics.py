"""Strict prediction alignment, grouped uncertainty and cue complementarity.

Reported intervals resample the supplied groups, not training seeds. Every
bootstrap draw uses the same groups for both systems in a paired comparison.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_curve,
)
from .manifests import binary


def probabilities(values) -> np.ndarray:
    p = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(
            "Probabilities must be finite and lie in [0,1]; no sanitization"
        )
    return p


def checked_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "group_id", "label", "p_fake"}
    if not required <= set(frame):
        raise ValueError(f"Missing prediction columns {sorted(required - set(frame))}")
    out = frame.copy()
    if out.empty:
        raise ValueError("Empty prediction population")
    for key in ("sample_id", "group_id"):
        if out[key].isna().any() or out[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {key}")
        out[key] = out[key].astype(str)
    if out.sample_id.duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    out["label"], out["p_fake"] = binary(out.label), probabilities(out.p_fake)
    return out.sort_values("sample_id").reset_index(drop=True)


def align(reference: pd.DataFrame, other: pd.DataFrame):
    a, b = checked_predictions(reference), checked_predictions(other)
    if not a.sample_id.equals(b.sample_id):
        raise ValueError(
            "Prediction populations differ; missing rows are not silently dropped"
        )
    if not a.label.equals(b.label) or not a.group_id.equals(b.group_id):
        raise ValueError("Label or group identity mismatch")
    return a, b


def choose_threshold(y, p, policy: str = "balanced_accuracy") -> float:
    if policy not in {"balanced_accuracy", "youden"}:
        raise ValueError("Threshold policy must be balanced_accuracy or youden")
    y, p = binary(y), probabilities(p)
    if len(y) != len(p) or len(np.unique(y)) != 2:
        raise ValueError("Calibration requires aligned predictions and both classes")
    order = np.argsort(p, kind="stable")
    yy, pp = y[order], p[order]
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, p, np.nextafter(1.0, 2.0)])
    i = np.searchsorted(pp, candidates, side="left")
    cumulative = np.r_[0, np.cumsum(yy)]
    fn, tn = cumulative[i], i - cumulative[i]
    ba = 0.5 * ((y.sum() - fn) / y.sum() + tn / (len(y) - y.sum()))
    return float(candidates[np.flatnonzero(ba == ba.max())[0]])


def equal_error_point(y, p) -> dict:
    """Piecewise-linear empirical ROC crossing of FPR and FNR.

    This target-label diagnostic is not an operating threshold or calibration.
    """
    y, p = binary(y), probabilities(p)
    if len(y) != len(p) or not len(y):
        raise ValueError("EER requires nonempty aligned scores")
    if len(np.unique(y)) != 2:
        return {"eer": None, "eer_threshold": None}
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    # The all-negative endpoint needs a finite score-domain threshold for JSON.
    thresholds[0] = np.nextafter(float(p.max()), np.inf)
    difference = fpr + tpr - 1
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0:
        return {"eer": float(fpr[right]), "eer_threshold": float(thresholds[right])}
    left = right - 1
    fraction = -difference[left] / (difference[right] - difference[left])
    return {
        "eer": float(fpr[left] + fraction * (fpr[right] - fpr[left])),
        "eer_threshold": float(thresholds[left] + fraction * (thresholds[right] - thresholds[left])),
    }


def equal_error_rate(y, p) -> float | None:
    return equal_error_point(y, p)["eer"]


def summary(y, p, threshold: float = 0.5) -> dict:
    y, p = binary(y), probabilities(p)
    if (
        len(y) != len(p)
        or not len(y)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= np.nextafter(1.0, 2.0)
    ):
        raise ValueError("Invalid metric inputs")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        float(np.mean(bins == b))
        * abs(float(y[bins == b].mean()) - float(p[bins == b].mean()))
        for b in range(10)
        if (bins == b).any()
    )
    return {
        "n": len(y),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        **equal_error_point(y, p),
        "eer_method": "piecewise-linear ROC crossing FPR=FNR; diagnostic only",
        "eer_status": "defined" if both else "undefined: one class",
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "f1_policy": "fake is positive; zero division returns zero",
        "accuracy": float((pred == y).mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_equal_width": ece,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_normalized": [
            [float(tn / (tn + fp)), float(fp / (tn + fp))] if tn + fp else [None, None],
            [float(fn / (fn + tp)), float(tp / (fn + tp))] if fn + tp else [None, None],
        ],
        "confusion_axes": "rows true, columns predicted; [real, fake]",
        "fpr": float(fp / (tn + fp)) if tn + fp else None,
        "tpr": float(tp / (tp + fn)) if tp + fn else None,
        "auc_status": "defined" if both else "undefined: one class, not zero AUC",
    }


def grouped_auc_interval(
    frame: pd.DataFrame,
    other: pd.DataFrame | None = None,
    *,
    draws: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    if draws < 20 or not 0 < confidence < 1:
        raise ValueError("At least 20 draws and 0<confidence<1 are required")
    a = checked_predictions(frame)
    b = align(a, other)[1] if other is not None else None
    if a.label.nunique() != 2:
        raise ValueError("AUC interval requires both classes")
    groups, inverse = np.unique(a.group_id, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("At least two independent sampling groups are required")
    rng = np.random.default_rng(seed)
    y, p = a.label.to_numpy(), a.p_fake.to_numpy()
    q = None if b is None else b.p_fake.to_numpy()
    values = []
    for _ in range(draws):
        counts = np.bincount(
            rng.integers(0, len(groups), len(groups)), minlength=len(groups)
        )
        w = counts[inverse]
        if not w[y == 0].sum() or not w[y == 1].sum():
            continue
        value = float(roc_auc_score(y, p, sample_weight=w))
        if q is not None:
            value = float(roc_auc_score(y, q, sample_weight=w)) - value
        values.append(value)
    if len(values) < 0.8 * draws:
        raise ValueError(
            "Too many one-class bootstrap draws; report a different justified grouping/design"
        )
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    point = float(roc_auc_score(y, p))
    if q is not None:
        point = float(roc_auc_score(y, q)) - point
    return {
        "estimate": point,
        "lower": float(lo),
        "upper": float(hi),
        "confidence": confidence,
        "requested_draws": draws,
        "valid_draws": len(values),
        "groups": len(groups),
        "seed": seed,
        "quantity": "other_minus_reference_auc" if b is not None else "auc",
        "method": "paired percentile cluster bootstrap",
        "limitation": "Conditional on these trained checkpoints and supplied groups; not across-seed uncertainty or causal evidence.",
    }


def complementarity(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    reference_threshold: float,
    other_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    a, b = align(reference, other)
    # Validate threshold ranges using the same metric contract.
    summary(a.label, a.p_fake, reference_threshold)
    summary(b.label, b.p_fake, other_threshold)
    ea = (a.p_fake.to_numpy() >= reference_threshold) != a.label.to_numpy()
    eb = (b.p_fake.to_numpy() >= other_threshold) != b.label.to_numpy()
    groups = np.select(
        [ea & ~eb, ~ea & eb, ea & eb],
        ["repaired_by_other", "regressed_by_other", "both_wrong"],
        default="both_correct",
    )
    cases = a[["sample_id", "group_id", "label"]].copy()
    cases["category"] = groups
    counts = {
        k: int((groups == k).sum())
        for k in (
            "repaired_by_other",
            "regressed_by_other",
            "both_wrong",
            "both_correct",
        )
    }
    return {
        "n": len(a),
        **counts,
        "reference_threshold": reference_threshold,
        "other_threshold": other_threshold,
        "other_correct_given_reference_wrong": float((ea & ~eb).sum() / ea.sum())
        if ea.sum()
        else None,
        "oracle_accuracy_upper_bound_not_deployable": float(1 - (ea & eb).mean()),
    }, cases


def aggregate_videos(frame: pd.DataFrame) -> pd.DataFrame:
    frame = checked_predictions(frame)
    if (
        "video_id" not in frame
        or frame.video_id.isna().any()
        or frame.video_id.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Complete video IDs are required")
    if (
        frame.groupby("video_id").label.nunique().gt(1).any()
        or frame.groupby("video_id").group_id.nunique().gt(1).any()
    ):
        raise ValueError("Conflicting labels or sampling groups within a video")
    result = (
        frame.groupby("video_id", sort=True)
        .agg(
            label=("label", "first"),
            group_id=("group_id", "first"),
            p_fake=("p_fake", "mean"),
            n_frames=("sample_id", "size"),
        )
        .reset_index()
    )
    result["sample_id"] = result.video_id
    return checked_predictions(result)


def generator_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    frame = checked_predictions(frame)
    if not {"generator", "source_domain"} <= set(frame):
        raise ValueError("Generator and source_domain annotations are required")
    for c in ("generator", "source_domain"):
        if (
            frame[c].isna().any()
            or frame[c].astype(str).str.strip().isin(["", "unknown"]).any()
        ):
            raise ValueError(f"Incomplete {c} annotations")
    rows = []
    for (domain, generator), fake in frame.loc[frame.label.eq(1)].groupby(
        ["source_domain", "generator"], sort=True
    ):
        real = frame.loc[frame.label.eq(0) & frame.source_domain.eq(domain)]
        pair = pd.concat([real, fake])
        rows.append(
            {
                "source_domain": str(domain),
                "generator": str(generator),
                "real_reference": "all real rows in same source_domain",
                **summary(pair.label, pair.p_fake, threshold),
            }
        )
    valid = [r["auc"] for r in rows if r["auc"] is not None]
    return {
        "groups": rows,
        "macro_auc": float(np.mean(valid)) if valid else None,
        "worst_group_auc": float(min(valid)) if valid else None,
        "auc_defined_groups": len(valid),
        "total_groups": len(rows),
        "limitation": "Groups reuse real references and are not independent replications. Missing real references yield undefined AUC.",
    }


def subgroup_metrics(frame, threshold, *, column="generator",
                     real_reference_policy="source_matched", draws=0, seed=42, confidence=0.95):
    """Explicit fake subgroup versus reviewed real-reference policy."""
    frame = checked_predictions(frame)
    if column not in {"generator", "paradigm"} or column not in frame:
        raise ValueError("A generator or paradigm annotation is required")
    if real_reference_policy not in {"source_matched", "pooled_all_real"}:
        raise ValueError("Declare source_matched or pooled_all_real references")
    if real_reference_policy == "source_matched":
        if "source_domain" not in frame:
            raise ValueError("Source-matched AUC requires reviewed source_domain annotations")
        for value in frame.source_domain:
            if pd.isna(value) or str(value).strip() in {"", "unknown"}:
                raise ValueError("Incomplete source_domain annotations")
        keys = ["source_domain", column]
    else:
        keys = [column]
    fake_frame = frame.loc[frame.label.eq(1)]
    if fake_frame[column].isna().any() or fake_frame[column].astype(str).str.strip().isin(["", "unknown"]).any():
        raise ValueError(f"Incomplete {column} annotations")
    rows = []
    for key, fake in fake_frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        real = frame.loc[frame.label.eq(0)]
        if real_reference_policy == "source_matched":
            real = real.loc[real.source_domain.eq(key[0])]
        pair = pd.concat([real, fake], ignore_index=True)
        row = {name: str(value) for name, value in zip(keys, key)}
        row.update(real_reference=real_reference_policy, **summary(pair.label, pair.p_fake, threshold))
        if draws and row["auc"] is not None:
            row["auc_interval"] = grouped_auc_interval(pair, draws=draws, seed=seed, confidence=confidence)
        rows.append(row)
    aucs = [row["auc"] for row in rows if row["auc"] is not None]
    return {
        "groups": rows, "real_reference_policy": real_reference_policy,
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "worst_group_auc": min(aucs) if aucs else None,
        "auc_defined_groups": len(aucs), "total_groups": len(rows),
        "limitation": "Real references are reused; subgroup estimates are dependent. Pooled references may confound source domains.",
    }
