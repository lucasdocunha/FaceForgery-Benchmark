"""Render existing numeric attributions in their own coordinate systems."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .contracts import contained, sha256
from .prepare import read_plan


def panels(encoded, rgb, attribution, mode):
    a = attribution[0] if attribution.ndim == 4 else attribution
    if a.ndim != 3 or a.shape[-2:] != encoded.shape[-2:]:
        raise ValueError("Attribution and input grids differ")
    global_layer = a.shape[0] == 1 and encoded.shape[0] > 1
    def channel_map(indices):
        return a[0] if global_layer else a[indices].sum(axis=0)
    result = []
    if mode in {"none", "concat", "concat_frequency"}:
        result.append(("RGB / spatial pixels", rgb.transpose(1, 2, 0), channel_map([0, 1, 2]), "spatial"))
        start = 3
    else:
        start = 0
    labels = {"magnitude": ["log magnitude"], "phase": ["phase"], "complex": ["real component", "imaginary component"],
              "frequency_3": ["high-pass magnitude"], "concat": ["log magnitude"],
              "concat_frequency": ["log magnitude", "phase", "high-pass", "low-pass"]}
    for offset, c in enumerate(range(start, encoded.shape[0])):
        view = np.clip((encoded[c] + 1) / 2, 0, 1)
        result.append((f"FFT {labels[mode][offset]} / frequency bins", view, channel_map([c]), "frequency"))
    return result


def _draw(ax, panel):
    title, view, heat, domain = panel
    if view.ndim == 2:
        ax.imshow(view, cmap="gray", vmin=0, vmax=1, origin="upper")
    else:
        ax.imshow(np.clip(view, 0, 1), origin="upper")
    scale = float(np.max(np.abs(heat)))
    normalized = heat / scale if scale else np.zeros_like(heat)
    artist = ax.imshow(normalized, cmap="RdBu_r", vmin=-1, vmax=1, alpha=.55, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (pixels)" if domain == "spatial" else "u (shifted FFT bin)", fontsize=8)
    ax.set_ylabel("y (pixels)" if domain == "spatial" else "v (shifted FFT bin)", fontsize=8)
    ax.tick_params(labelsize=7)
    return artist


def render_cell(encoded, rgb, attribution, mode, method, output, label=""):
    parts = panels(encoded, rgb, attribution, mode)
    fig, axes = plt.subplots(1, len(parts), figsize=(4 * len(parts), 4.7), squeeze=False, constrained_layout=True)
    for ax, panel in zip(axes[0], parts):
        artist = _draw(ax, panel)
        fig.colorbar(artist, ax=ax, fraction=.035, label="map / maximum absolute value")
    semantics = "nonnegative localization/attention" if method in {"gradcam", "attention_rollout"} else "signed encoded-input map"
    fig.suptitle(f"{method} | {semantics}\n{label}", fontsize=10)
    fig.supxlabel("Display normalized independently; use raw arrays for quantitative analysis.", fontsize=8)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)


def _load_cell(root, task, model, method, name):
    directory = root / task / model / method
    record_path = directory / f"{name}.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if record["status"] != "complete":
        return None
    for filename, checksum in record["artifacts"].items():
        if Path(filename).name != filename or sha256(directory / filename) != checksum:
            raise ValueError("Cell artifact checksum mismatch")
    asset = contained(root, record["asset"])
    if sha256(asset) != record["identity"]["asset_sha256"]:
        raise ValueError("Input asset checksum mismatch")
    with np.load(asset, allow_pickle=False) as bundle:
        encoded, rgb = bundle["encoded_input"], bundle["rgb"]
    with np.load(directory / f"{name}.npz", allow_pickle=False) as bundle:
        attr = bundle["attribution"]
    return record, encoded, rgb, attr


def compare(plan_path, output):
    plan, root = read_plan(plan_path), Path(output)
    roster = plan["cohorts"]["task2"]["roster"]
    if len(roster) != 2 or plan["models"][roster[0]]["mode"] != "none":
        raise ValueError("Task 2 requires a declared RGB/frequency pair")
    boards = root / "task2/comparisons"; boards.mkdir(parents=True, exist_ok=True)
    status = {"plan_id": plan["plan_id"], "complete": [], "pending_or_unsupported": []}
    for method in plan["config"]["methods"]:
        for row in plan["cohorts"]["task2"]["samples"]:
            for rep in range(int(plan["config"]["settings"].get("replicates", 1))):
                name = f"{row['id']}_r{rep}"
                pair = [_load_cell(root, "task2", model, method, name) for model in roster]
                key = f"{method}/{name}"
                if any(item is None for item in pair):
                    status["pending_or_unsupported"].append(key); continue
                left, right = pair
                if (left[0]["identity"]["plan_id"] != plan["plan_id"]
                    or right[0]["identity"]["plan_id"] != plan["plan_id"]
                    or left[0]["identity"]["image_sha256"] != right[0]["identity"]["image_sha256"]):
                    raise ValueError("Task 2 cells do not refer to the same frozen image")
                left_parts = panels(*left[1:], "none")
                right_parts = panels(*right[1:], plan["models"][roster[1]]["mode"])
                n = len(right_parts)
                fig = plt.figure(figsize=(9, max(4.7, 3.5 * n)), constrained_layout=True)
                gs = fig.add_gridspec(n, 2)
                ax = fig.add_subplot(gs[:, 0]); artist = _draw(ax, left_parts[0])
                fig.colorbar(artist, ax=ax, fraction=.035)
                for i, panel in enumerate(right_parts):
                    ax = fig.add_subplot(gs[i, 1]); artist = _draw(ax, panel)
                    fig.colorbar(artist, ax=ax, fraction=.035)
                fig.suptitle(f"{method} | ID {row['id']} | truth={row['y_true']}\n"
                    f"RGB p={left[0]['fresh_probability']:.3f} versus FFT p={right[0]['fresh_probability']:.3f}", fontsize=11)
                fig.supxlabel("Different coordinate systems; each map independently scaled. FFT bins are not facial locations.", fontsize=8)
                destination = boards / method; destination.mkdir(exist_ok=True)
                fig.savefig(destination / f"{name}.png", dpi=180)
                fig.savefig(destination / f"{name}.pdf")
                plt.close(fig); status["complete"].append(key)
    (boards / "coverage.json").write_text(json.dumps(status, indent=2) + "\n")
    return {"complete": len(status["complete"]), "pending_or_unsupported": len(status["pending_or_unsupported"])}
