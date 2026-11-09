"""Pure-data contracts: no image substitution, implicit joins, or test calibration."""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score)

HF_REPO = "lucasoc/MFFI-Models"
HF_REVISION = "3bef179cdc08850e1d720182e55a040d00c9d582"
FAMILIES = ("resnet", "xception", "mobilenet", "vit", "clip", "dino")
SEEDS = (42, 123, 2024)
PURE_FREQUENCY = ("magnitude", "phase", "complex", "frequency_3")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def freeze(path: str | Path, value: Any) -> None:
    """Atomic create-or-verify. Never overwrite an existing experiment contract."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name); f.write(data)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Frozen artifact differs: {path}; use a new output directory")
    finally:
        temporary.unlink(missing_ok=True)


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError(f"Unsafe identifier: {name!r}")
    return name


def contained(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path escapes root")
    return resolved


def binary(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all() or not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} must be a finite binary vector")
    return arr.astype(np.int64)


def probability(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    return arr


def manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if "img_name" not in frame or "label" not in frame:
        raise ValueError("Manifest requires img_name,label columns")
    if frame["img_name"].isna().any():
        raise ValueError("Missing image name")
    names = frame["img_name"].astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError("Empty or duplicate image names")
    for name in names:
        contained(Path("/manifest-root"), name)
    return pd.DataFrame({"id": np.arange(len(frame), dtype=np.int64),
                         "img_name": names, "y_true": binary(frame["label"], "label")})


def raw_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path); frame.columns = frame.columns.str.strip()
    required = {"id", "y_true", "prob_pos"}
    if not required.issubset(frame):
        raise ValueError(f"Missing prediction columns: {required - set(frame.columns)}")
    ids = pd.to_numeric(frame["id"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(ids).all() or (ids != np.floor(ids)).any()
            or (ids < 0).any() or (ids > 2**53 - 1).any()):
        raise ValueError("Legacy IDs must be exact nonnegative integers")
    out = pd.DataFrame({"id": ids.astype(np.int64),
                        "y_true": binary(frame["y_true"], "y_true"),
                        "prob_pos": probability(frame["prob_pos"])})
    if "y_pred" in frame:
        out["released_y_pred"] = binary(frame["y_pred"], "y_pred")
    return out


def align_predictions(frame: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Exact ID-set and label checks; never an implicit inner join."""
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    if expected["id"].duplicated().any():
        raise ValueError("Duplicate manifest IDs")
    if set(frame.id) != set(expected.id):
        raise ValueError("Prediction coverage differs from the complete expected population")
    out = frame.set_index("id").loc[expected.id].reset_index()
    if not np.array_equal(out.y_true.to_numpy(), expected.y_true.to_numpy()):
        raise ValueError("Prediction/manifest label disagreement")
    out["img_name"] = expected.img_name.to_numpy()
    return out


def curate_validation(frames: dict[str, pd.DataFrame], expected: pd.DataFrame,
                      *, allow_identical_duplicates: bool = False):
    """Explicit derived view. Missing observations are documented, never imputed."""
    if not frames:
        raise ValueError("Empty validation roster")
    cleaned, reports = {}, {}
    population = set(expected.id)
    for name, frame in sorted(frames.items()):
        duplicate_ids = sorted(frame.loc[frame.id.duplicated(False), "id"].unique().tolist())
        missing = sorted(population - set(frame.id))
        if set(frame.id) - population:
            raise ValueError("Unexpected validation IDs")
        if (duplicate_ids or missing) and not allow_identical_duplicates:
            raise ValueError("Validation coverage defect; explicit curation is required")
        for ident in duplicate_ids:
            rows = frame.loc[frame.id.eq(ident)]
            if any(rows[column].nunique(dropna=False) != 1 for column in rows.columns):
                raise ValueError("Non-identical duplicate validation predictions")
        clean = frame.drop_duplicates("id", keep="first").copy()
        cleaned[name] = clean
        reports[name] = {"original_rows": len(frame), "unique_rows": len(clean),
                         "duplicate_ids": duplicate_ids, "missing_ids": missing}
    observed = set(next(iter(cleaned.values())).id)
    if any(set(frame.id) != observed for frame in cleaned.values()):
        raise ValueError("Inconsistent validation populations across models")
    derived = expected.loc[expected.id.isin(observed)].reset_index(drop=True)
    return {name: align_predictions(frame, derived) for name, frame in cleaned.items()}, derived, reports


def threshold(y, p) -> float:
    """Validation balanced accuracy, O(N log N), smallest-candidate tie break."""
    y, p = binary(y, "validation labels"), probability(p)
    if len(y) != len(p) or len(np.unique(y)) != 2:
        raise ValueError("Calibration requires both classes and aligned predictions")
    order = np.argsort(p, kind="stable"); sorted_p, sorted_y = p[order], y[order]
    candidates = np.unique(np.r_[0., .5, 1., p])
    positions = np.searchsorted(sorted_p, candidates, side="left")
    positives = np.r_[0, np.cumsum(sorted_y)]
    fn = positives[positions]; tn = positions - fn
    score = .5 * ((y.sum() - fn) / y.sum() + tn / (len(y) - y.sum()))
    return float(candidates[np.flatnonzero(score == score.max())[0]])


def metrics(y, p, t: float) -> dict:
    y, p = binary(y, "labels"), probability(p)
    if len(y) != len(p) or len(np.unique(y)) != 2 or not 0 <= t <= 1:
        raise ValueError("Invalid evaluation inputs")
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return {"n": len(y), "threshold": t, "auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "brier": float(brier_score_loss(y, p)), "ece_10_equal_width": float(ece),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fpr": float(fp / (tn + fp)), "fnr": float(fn / (tp + fn))}


def aligned_roster(tables: dict[str, pd.DataFrame]) -> tuple[list[str], pd.DataFrame]:
    if not tables:
        raise ValueError("Empty model roster")
    names = sorted(tables)
    reference = tables[names[0]].sort_values("id").reset_index(drop=True)
    for name in names:
        frame = tables[name].sort_values("id").reset_index(drop=True)
        if frame.id.duplicated().any() or not frame[["id", "y_true", "img_name"]].equals(reference[["id", "y_true", "img_name"]]):
            raise ValueError("Model roster is not completely aligned")
        binary(frame.y_pred, "frozen decisions")
    return names, reference


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"))


def strata(frame: pd.DataFrame) -> np.ndarray:
    y, pred = binary(frame.y_true, "labels"), binary(frame.y_pred, "decisions")
    return np.where(y == 1, np.where(pred == 1, "TP", "FN"), np.where(pred == 0, "TN", "FP"))


def sample64(tables: dict[str, pd.DataFrame], reference: str, seed: int = 42) -> pd.DataFrame:
    aligned_roster(tables)
    if reference not in tables:
        raise ValueError("Reference checkpoint is absent")
    frame = tables[reference].sort_values("id").reset_index(drop=True).copy()
    frame["reference_stratum"] = strata(frame)
    chosen = []
    for name in ("TP", "FN", "TN", "FP"):
        candidates = frame.loc[frame.reference_stratum.eq(name)]
        if len(candidates) < 16:
            raise ValueError(f"Need 16 {name}, found {len(candidates)}")
        chosen.extend(_rng(seed, name).choice(candidates.index, 16, replace=False).tolist())
    result = frame.loc[chosen].reset_index(drop=True)
    assert len(result) == 64 and result.id.nunique() == 64
    return result


def shared_errors(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names, ref = aligned_roster(tables)
    bad = np.ones(len(ref), dtype=bool)
    for name in names:
        frame = tables[name].sort_values("id")
        bad &= frame.y_true.to_numpy() != frame.y_pred.to_numpy()
    return ref.loc[bad, ["id", "img_name", "y_true"]].reset_index(drop=True)


def matched_controls(tables: dict[str, pd.DataFrame], hard: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    _, ref = aligned_roster(tables)
    eligible = ref.loc[~ref.id.isin(hard.id)]
    chosen = []
    for label in (0, 1):
        n = int(hard.y_true.eq(label).sum())
        pool = eligible.loc[eligible.y_true.eq(label)]
        if len(pool) < n:
            raise ValueError("Insufficient label-matched controls without replacement")
        chosen.extend(_rng(seed, f"controls-{label}").choice(pool.index, n, replace=False).tolist())
    return ref.loc[chosen, ["id", "img_name", "y_true"]].reset_index(drop=True)


def select_frequency(table: pd.DataFrame, floor: float = .60) -> dict:
    required = {"model_family", "fourier_mode", "regime", "seed", "split", "auc"}
    if not required.issubset(table) or not table["split"].eq("val").all():
        raise ValueError("Selection requires validation-only aggregate metrics")
    if table[list(required - {"auc"})].duplicated().any():
        raise ValueError("Duplicate validation configuration")
    if not np.isfinite(table.auc.to_numpy(float)).all() or not table.auc.between(0, 1).all():
        raise ValueError("Invalid validation AUC")
    ranking, excluded = [], []
    for (family, regime), rows in table.groupby(["model_family", "regime"]):
        rgb = rows.loc[rows.fourier_mode.eq("none")].set_index("seed")
        for mode in PURE_FREQUENCY:
            spectral = rows.loc[rows.fourier_mode.eq(mode)].set_index("seed")
            key = f"{family}/{mode}/{regime}"
            if set(rgb.index) != set(SEEDS) or set(spectral.index) != set(SEEDS):
                excluded.append({"pair": key, "reason": "incomplete paired seed roster"}); continue
            a, b = rgb.loc[list(SEEDS), "auc"].to_numpy(), spectral.loc[list(SEEDS), "auc"].to_numpy()
            if min(a.min(), b.min()) < floor:
                excluded.append({"pair": key, "reason": "below prespecified AUC floor"}); continue
            ranking.append({"model_family": family, "fourier_mode": mode, "regime": regime,
                            "worst_domain_auc": float(np.minimum(a, b).mean()),
                            "positive_drop": float(np.maximum(a - b, 0).mean()),
                            "rgb_auc": float(a.mean()), "frequency_auc": float(b.mean())})
    if not ranking:
        raise ValueError("No eligible RGB/frequency pair")
    ranking.sort(key=lambda r: (-r["worst_domain_auc"], r["positive_drop"], r["model_family"], r["fourier_mode"], r["regime"]))
    return {"selection_split": "val", "seeds": list(SEEDS), "auc_floor": floor,
            "selected": ranking[0], "ranking": ranking, "excluded": excluded,
            "evidence": "released aggregate validation metrics, not image inference reproduction"}
