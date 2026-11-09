"""Small synthetic contract checks, not benchmark experiments or training."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from src.explicability.contracts import (align_predictions, curate_validation, freeze,
    matched_controls, sample64, select_frequency, shared_errors, threshold)
from src.explicability.prepare import prepare, read_plan
from src.explicability.report import coverage, export_tables


class ContractTests(unittest.TestCase):
    def population(self, n=96):
        y = np.repeat([1, 1, 0, 0], n // 4)
        pred = np.repeat([1, 0, 0, 1], n // 4)
        return pd.DataFrame({"id": range(n), "img_name": [f"{i}.png" for i in range(n)],
            "y_true": y, "y_pred": pred, "prob_pos": np.where(pred, .95, .05)})

    def test_exact_cohort_deterministic_under_row_permutation(self):
        a = self.population()
        b = a.copy(); b["y_pred"] = 1 - b.y_pred
        first = sample64({"a": a, "b": b}, "a")
        second = sample64({"b": b.sample(frac=1, random_state=3), "a": a.iloc[::-1]}, "a")
        self.assertEqual(first.id.tolist(), second.id.tolist())
        self.assertEqual(first.reference_stratum.value_counts().to_dict(), {"TP": 16, "FN": 16, "TN": 16, "FP": 16})
        self.assertEqual(first.id.nunique(), 64)

    def test_short_stratum_is_error(self):
        frame = self.population(32)
        with self.assertRaises(ValueError): sample64({"a": frame}, "a")

    def test_missing_prediction_is_not_shared_failure(self):
        frame = self.population()
        with self.assertRaises(ValueError): shared_errors({"a": frame, "b": frame.iloc[:-1]})

    def test_empty_intersection_and_controls(self):
        a = self.population(); b = a.copy(); b["y_pred"] = 1 - b.y_pred
        hard = shared_errors({"a": a, "b": b})
        self.assertEqual(len(hard), 0)
        self.assertEqual(len(matched_controls({"a": a, "b": b}, hard)), 0)

    def test_shared_failure_and_label_matching(self):
        a = self.population(); hard = shared_errors({"a": a, "b": a.copy()})
        controls = matched_controls({"a": a, "b": a.copy()}, hard)
        self.assertFalse(set(controls.id) & set(hard.id))
        self.assertEqual(controls.y_true.value_counts().to_dict(), hard.y_true.value_counts().to_dict())

    def test_strict_label_and_duplicate_checks(self):
        a = self.population()
        with self.assertRaises(ValueError): align_predictions(pd.concat([a, a.iloc[:1]]), a)
        b = a.copy(); b.loc[0, "y_true"] = 0
        with self.assertRaises(ValueError): align_predictions(b, a)

    def test_curation_is_explicit_and_consistent(self):
        a = self.population(); broken = pd.concat([a.iloc[1:], a.iloc[[1]]], ignore_index=True)
        with self.assertRaises(ValueError): curate_validation({"a": broken}, a)
        clean, derived, audit = curate_validation({"a": broken}, a, allow_identical_duplicates=True)
        self.assertEqual(len(derived), len(a) - 1)
        self.assertEqual(audit["a"]["missing_ids"], [0])
        self.assertEqual(audit["a"]["duplicate_ids"], [1])
        other = broken.copy(); other.loc[len(other)-1, "prob_pos"] = .4
        with self.assertRaises(ValueError): curate_validation({"a": other}, a, allow_identical_duplicates=True)

    def test_calibration_matches_brute_force(self):
        rng = np.random.default_rng(123)
        y = rng.integers(0, 2, 200); p = rng.choice(np.linspace(0, 1, 20), 200)
        candidates = np.unique(np.r_[0, .5, 1, p])
        scores = [balanced_accuracy_score(y, p >= t) for t in candidates]
        self.assertEqual(threshold(y, p), candidates[np.argmax(scores)])

    def test_selection_rejects_test_metrics_and_floor_effect(self):
        rows = []
        for family, rgb, freq in (("resnet", .85, .75), ("vit", .51, .51)):
            for seed in (42, 123, 2024):
                for mode, auc in (("none", rgb), ("magnitude", freq)):
                    rows.append(dict(model_family=family, regime="finetune", seed=seed, fourier_mode=mode, split="val", auc=auc))
        table = pd.DataFrame(rows)
        self.assertEqual(select_frequency(table)["selected"]["model_family"], "resnet")
        table.loc[0, "split"] = "test"
        with self.assertRaises(ValueError): select_frequency(table)

    def test_freeze_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "contract.json"
            freeze(p, {"x": 1}); freeze(p, {"x": 1})
            with self.assertRaises(ValueError): freeze(p, {"x": 2})

    def test_prepare_report_and_table_export_on_synthetic_data(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); models = root / "models"; out = root / "out"
            test = self.population()
            val = self.population(48); val["prob_pos"] = np.where(val.y_true, .9, .1)
            for name, frame in (("test", test), ("val", val)):
                frame[["img_name", "y_true"]].rename(columns={"y_true": "label"}).to_csv(root / f"{name}.csv", index=False)
            for seed in (42, 123, 2024):
                results = models / f"resnet/none/finetune/seed_{seed}/results"
                results.mkdir(parents=True)
                (results / "run_config.json").write_text("{}")
                for name, frame in (("val", val), ("test", test)):
                    frame[["id", "y_true", "prob_pos"]].to_csv(results / f"predictions_{name}.csv", index=False)
            cfg = {"version": 1, "seed": 42, "regime": "finetune", "primary_seed": 42,
                "families": ["resnet"], "methods": ["integrated_gradients"],
                "settings": {"baseline": "encoded_zero", "replicates": 1}, "task2": {"enabled": False}}
            (root / "config.yaml").write_text(yaml.safe_dump(cfg))
            plan = prepare(root / "config.yaml", models, root / "val.csv", root / "test.csv", out,
                           acknowledge_legacy_row_ids=True)
            self.assertEqual(len(plan["cohorts"]["task1"]["samples"]), 64)
            self.assertEqual(read_plan(out / "plan.json")["plan_id"], plan["plan_id"])
            result = coverage(out / "plan.json", out)
            self.assertEqual(result["tasks"]["task1"]["missing"], 64)
            self.assertGreater(export_tables(out), 0)
            altered = json.loads((out / "plan.json").read_text()); altered["reference_model"] = "other"
            (out / "plan.json").write_text(json.dumps(altered))
            with self.assertRaises(ValueError): read_plan(out / "plan.json")


if __name__ == "__main__": unittest.main()
