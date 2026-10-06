#!/usr/bin/env python3
"""Evaluate SupCon and Graph (GAT) models on the MFFI test set."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

repo_dir = Path(__file__).resolve().parents[1]
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from src.experimental.features import extract_features, open_cache
from src.experimental.feature_training import predict_metric_run
from src.experimental.graph_training import predict_graph_run
from src.robustness.statistics import summary
from src.robustness.provenance import write_json, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1", help="Device for feature extraction and inference")
    parser.add_argument("--split", default="test", choices=["test", "test_d"], help="Split to evaluate (test or test_d)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-gat", action="store_true", help="Only evaluate SupCon")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    dino_checkpoint = Path("/media/ssd2/lucas.ocunha/models-tcc/dino/srm/finetune_robust/seed_42/weights/best.pth")
    test_manifest = repo_dir / "data/manifests/test.csv"
    if args.split == "test_d":
        test_root = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")
        cache_dir = Path("/media/ssd2/lucas.ocunha/models-tcc/experimental/graphs/feature_cache/test_d")
    else:
        test_root = repo_dir / "data/datasets/phase1/testset"
        cache_dir = Path("/media/ssd2/lucas.ocunha/models-tcc/experimental/graphs/feature_cache/test")

    cache_dir.mkdir(parents=True, exist_ok=True)
    supcon_dir = Path("/media/ssd2/lucas.ocunha/models-tcc/experimental/graphs/metric_supcon_seed_42")
    gat_dir = Path("/media/ssd2/lucas.ocunha/models-tcc/experimental/graphs/graph_gat_seed_42")

    print("=" * 80)
    print(f"🚀 AVALIAÇÃO NO CONJUNTO DE TESTE ({args.split.upper()} - 181.947 amostras)")
    print(f"Device    : {args.device}")
    print(f"Split     : {args.split}")
    print(f"Checkpoint: {dino_checkpoint}")
    print(f"Manifest  : {test_manifest}")
    print(f"Image Root: {test_root}")
    print(f"Cache dir : {cache_dir}")
    print("=" * 80)

    cache_dirs = [p for p in cache_dir.iterdir() if p.is_dir() and (p / "cache.json").is_file()]
    if cache_dirs:
        actual_cache_dir = cache_dirs[0]
        print(f"\n⏭️  [{time.strftime('%T')}] Cache de features de teste já existente em {actual_cache_dir}. Carregando...")
    else:
        print(f"\n▶️  [{time.strftime('%T')}] Extraindo cache de features DINO-SRM no conjunto de teste...")
        started = time.perf_counter()
        actual_cache_dir = extract_features(
            str(dino_checkpoint),
            str(test_manifest),
            str(test_root),
            str(cache_dir),
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            use_amp=True,
        )
        print(f"✅ Cache de teste extraído em {time.perf_counter() - started:.1f}s!")

    store = open_cache(actual_cache_dir)
    test_features = store.features
    test_labels = store.frame.label.to_numpy(dtype=np.int64)
    n_samples = len(test_labels)
    n_real = int((test_labels == 0).sum())
    n_fake = int((test_labels == 1).sum())
    print(f"📦 Total amostras de teste: {n_samples:,} (Reais: {n_real:,}, Fakes: {n_fake:,})")

    results_table = []

    # 2. Avaliação do SupCon (Seed 42)
    if supcon_dir.is_dir():
        print(f"\n▶️  [{time.strftime('%T')}] Avaliando SupCon (Seed 42) no conjunto de teste...")
        calib_file = supcon_dir / "calibration/calibration.json"
        if calib_file.is_file():
            calib = json.loads(calib_file.read_text())
            threshold = float(calib["frame_threshold"])
        else:
            calib_root = supcon_dir / "calibration.json"
            calib = json.loads(calib_root.read_text())
            threshold = float(calib["frame_threshold"])
        print(f"🔑 Threshold Youden congelado do val: {threshold:.6f}")

        t0 = time.perf_counter()
        p_fake_supcon = predict_metric_run(supcon_dir, test_features)
        supcon_time = time.perf_counter() - t0

        metrics_supcon = summary(test_labels, p_fake_supcon, threshold=threshold)
        metrics_supcon["evaluation_split"] = args.split
        metrics_supcon["threshold_frozen"] = threshold
        metrics_supcon["inference_seconds"] = supcon_time

        out_metrics = supcon_dir / f"{args.split}_metrics.json"
        write_json(out_metrics, metrics_supcon)
        print(f"✅ SupCon {args.split} metrics salvas em: {out_metrics}")

        tn, fp, fn, tp = (
            metrics_supcon["confusion_matrix"][0][0],
            metrics_supcon["confusion_matrix"][0][1],
            metrics_supcon["confusion_matrix"][1][0],
            metrics_supcon["confusion_matrix"][1][1],
        )
        spec = tn / (tn + fp) * 100
        sens = tp / (tp + fn) * 100
        results_table.append({
            "Modelo": "SupCon (Seed 42)",
            f"{args.split.upper()} AUC": f"{metrics_supcon['auc']*100:.2f}%",
            "Acurácia": f"{metrics_supcon['accuracy']*100:.2f}%",
            "F1-Score": f"{metrics_supcon['f1']*100:.2f}%",
            "TNR (Real Recall)": f"{spec:.2f}%",
            "TPR (Fake Recall)": f"{sens:.2f}%",
            "EER": f"{metrics_supcon['eer']*100:.2f}%",
        })

    # 3. Avaliação do GAT (Seed 42)
    if gat_dir.is_dir() and not args.skip_gat:
        print(f"\n▶️  [{time.strftime('%T')}] Avaliando GAT (Seed 42) no conjunto {args.split}...")
        calib_file = gat_dir / "calibration/calibration.json"
        if calib_file.is_file():
            calib = json.loads(calib_file.read_text())
            threshold = float(calib["frame_threshold"])
        else:
            calib_root = gat_dir / "calibration.json"
            calib = json.loads(calib_root.read_text())
            threshold = float(calib["frame_threshold"])
        print(f"🔑 Threshold Youden congelado do val: {threshold:.6f}")

        t0 = time.perf_counter()
        p_fake_gat = predict_graph_run(gat_dir, test_features, protocol="inductive")
        gat_time = time.perf_counter() - t0

        metrics_gat = summary(test_labels, p_fake_gat, threshold=threshold)
        metrics_gat["evaluation_split"] = args.split
        metrics_gat["threshold_frozen"] = threshold
        metrics_gat["inference_seconds"] = gat_time

        out_metrics = gat_dir / f"{args.split}_metrics.json"
        write_json(out_metrics, metrics_gat)
        print(f"✅ GAT {args.split} metrics salvas em: {out_metrics}")

        tn, fp, fn, tp = (
            metrics_gat["confusion_matrix"][0][0],
            metrics_gat["confusion_matrix"][0][1],
            metrics_gat["confusion_matrix"][1][0],
            metrics_gat["confusion_matrix"][1][1],
        )
        spec = tn / (tn + fp) * 100
        sens = tp / (tp + fn) * 100
        results_table.append({
            "Modelo": "GAT (Seed 42)",
            f"{args.split.upper()} AUC": f"{metrics_gat['auc']*100:.2f}%",
            "Acurácia": f"{metrics_gat['accuracy']*100:.2f}%",
            "F1-Score": f"{metrics_gat['f1']*100:.2f}%",
            "TNR (Real Recall)": f"{spec:.2f}%",
            "TPR (Fake Recall)": f"{sens:.2f}%",
            "EER": f"{metrics_gat['eer']*100:.2f}%",
        })

    # Imprime a tabela comparativa
    print("\n" + "=" * 80)
    print(f"📊 RESULTADOS NO CONJUNTO {args.split.upper()} (MFFI {args.split}):")
    print("=" * 80)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_markdown(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
