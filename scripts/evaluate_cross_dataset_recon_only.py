#!/usr/bin/env python3
"""
Avaliação Cross-Dataset (Celeb-DF v2 e DF-40) para a família Reconstruction (recon_only).
Avalia as 5 sementes canônicas (42, 123, 2024, 7, 2025) e gera ensemble de 5 sementes.
Salva métricas e saídas .npz / .csv no padrão do repositório em run_dir/results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.experimental.reconstruction.training import load_model
from src.pipelines.evaluation import binary_metrics, safe_auc

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


class CelebDFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, crops_dir: Path, image_size: int = 224):
        self.df = df
        self.crops_dir = crops_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.crops_dir / str(row["img_name"])
        with Image.open(img_path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, int(row["target"]), idx


class DF40Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(str(row["img_name"]))
        with Image.open(img_path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, int(row["target"]), idx


def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.inference_mode():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist())
            
    p = np.nan_to_num(np.array(all_probs), nan=0.5, posinf=1.0, neginf=0.0)
    t = np.array(all_targets, dtype=int)
    return p, t


def evaluate_celeb_df(
    model: torch.nn.Module,
    celeb_loader: DataLoader,
    df_meta: pd.DataFrame,
    threshold: float,
    results_dir: Path,
    device: torch.device,
    seed: int,
) -> dict:
    t0 = time.time()
    p_fakes, y_trues = run_inference(model, celeb_loader, device)
    dt = time.time() - t0

    # Frame metrics
    frame_auc = safe_auc(y_trues, p_fakes)
    frame_preds = (p_fakes >= threshold).astype(int)
    frame_acc = float(accuracy_score(y_trues, frame_preds))
    frame_f1 = float(f1_score(y_trues, frame_preds, zero_division=0))
    frame_p = float(precision_score(y_trues, frame_preds, zero_division=0))
    frame_r = float(recall_score(y_trues, frame_preds, zero_division=0))
    frame_spec = float(recall_score(1 - y_trues, 1 - frame_preds, zero_division=0))

    # Video metrics (aggregation by video_id)
    df_temp = df_meta.copy()
    df_temp["prob_fake"] = p_fakes
    video_df = df_temp.groupby("video_id").agg({"target": "first", "prob_fake": "mean"}).reset_index()

    vid_y = video_df["target"].to_numpy().astype(int)
    vid_p = np.nan_to_num(video_df["prob_fake"].to_numpy(), nan=0.5, posinf=1.0, neginf=0.0)
    vid_preds = (vid_p >= threshold).astype(int)

    video_auc = safe_auc(vid_y, vid_p)
    video_acc = float(accuracy_score(vid_y, vid_preds))
    video_f1 = float(f1_score(vid_y, vid_preds, zero_division=0))
    video_p = float(precision_score(vid_y, vid_preds, zero_division=0))
    video_r = float(recall_score(vid_y, vid_preds, zero_division=0))
    video_spec = float(recall_score(1 - vid_y, 1 - vid_preds, zero_division=0))

    # Save metrics
    results_dir.mkdir(parents=True, exist_ok=True)
    row_frame = {
        "model_family": "reconstruction", "variant": "recon_only",
        "seed": seed, "split": "celeb_df",
        "threshold": threshold, "auc": frame_auc, "acc": frame_acc,
        "f1": frame_f1, "precision": frame_p, "recall": frame_r,
        "specificity": frame_spec, "loss": 0.0,
        "frame_auc": frame_auc, "frame_acc": frame_acc, "frame_f1": frame_f1,
        "video_auc": video_auc, "video_acc": video_acc, "video_f1": video_f1,
    }
    pd.DataFrame([row_frame]).to_csv(results_dir / "metrics_celeb_df.csv", index=False)

    row_video = {
        "model_family": "reconstruction", "variant": "recon_only",
        "seed": seed, "split": "celeb_df_video",
        "threshold": threshold, "auc": video_auc, "acc": video_acc,
        "f1": video_f1, "precision": video_p, "recall": video_r,
        "specificity": video_spec, "loss": 0.0,
    }
    pd.DataFrame([row_video]).to_csv(results_dir / "metrics_celeb_df_video.csv", index=False)

    np.savez_compressed(
        results_dir / "outputs_celeb_df.npz",
        y_true=y_trues, p_fake=p_fakes, video_y=vid_y, video_p=vid_p, threshold=threshold
    )

    logger.info(
        f"[Seed {seed:4d}] Celeb-DF v2 -> Frame AUC: {frame_auc*100:.2f}% | Video AUC: {video_auc*100:.2f}% | Video Acc: {video_acc*100:.2f}% ({dt:.1f}s)"
    )

    return {
        "seed": seed, "frame_auc": frame_auc, "frame_acc": frame_acc, "frame_f1": frame_f1,
        "video_auc": video_auc, "video_acc": video_acc, "video_f1": video_f1,
        "p_fakes": p_fakes, "vid_p": vid_p, "y_trues": y_trues, "vid_y": vid_y
    }


def evaluate_df40(
    model: torch.nn.Module,
    df40_loader: DataLoader,
    manifest_df: pd.DataFrame,
    threshold: float,
    results_dir: Path,
    device: torch.device,
    seed: int,
) -> dict:
    t0 = time.time()
    p_fakes, y_trues = run_inference(model, df40_loader, device)
    dt = time.time() - t0

    # Overall metrics
    overall = binary_metrics(y_trues, p_fakes, threshold=threshold)
    auc = overall["auc"]
    acc = overall["acc"]
    f1 = overall["f1"]
    recall = overall["recall"]
    spec = overall["specificity"]

    # Save overall metrics
    results_dir.mkdir(parents=True, exist_ok=True)
    row_df40 = {
        "model_family": "reconstruction", "variant": "recon_only",
        "seed": seed, "split": "df40",
        "threshold": threshold, "loss": 0.0,
        "acc": acc, "precision": overall["precision"], "recall": recall,
        "f1": f1, "auc": auc, "specificity": spec,
        "tp": overall["tp"], "fp": overall["fp"], "fn": overall["fn"], "tn": overall["tn"],
    }
    pd.DataFrame([row_df40]).to_csv(results_dir / "metrics_df40.csv", index=False)

    # Save predictions
    pred_df = manifest_df.copy()
    pred_df["prob_fake"] = p_fakes
    pred_df["pred"] = (p_fakes >= threshold).astype(int)
    pred_df.to_csv(results_dir / "predictions_df40.csv", index=False)

    np.savez_compressed(
        results_dir / "outputs_df40.npz",
        probs=p_fakes, y_true=y_trues, y_pred=(p_fakes >= threshold).astype(int),
        threshold=threshold, ids=manifest_df.index.values
    )

    # Paradigm breakdown
    paradigms = {}
    real_mask = (y_trues == 0)
    for p_name in sorted(manifest_df["paradigm"].dropna().unique()):
        if p_name == "real":
            continue
        p_mask = (manifest_df["paradigm"] == p_name).values
        sub_mask = p_mask | real_mask
        sub_auc = safe_auc(y_trues[sub_mask], p_fakes[sub_mask])
        sub_rec = float(recall_score(y_trues[p_mask], (p_fakes[p_mask] >= threshold).astype(int), zero_division=0))
        paradigms[p_name] = {"auc": sub_auc, "recall": sub_rec}

    logger.info(
        f"[Seed {seed:4d}] DF-40        -> Overall AUC: {auc*100:.2f}% | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Recall: {recall*100:.2f}% ({dt:.1f}s)"
    )

    return {
        "seed": seed, "auc": auc, "acc": acc, "f1": f1, "recall": recall, "spec": spec,
        "p_fakes": p_fakes, "y_trues": y_trues, "paradigms": paradigms
    }


def main():
    parser = argparse.ArgumentParser(description="Avaliação Cross-Dataset do recon_only")
    parser.add_argument("--device", default="cuda:0", help="Dispositivo para inferência")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size de inferência")
    parser.add_argument("--workers", type=int, default=4, help="Número de workers no DataLoader")
    parser.add_argument("--seeds", default="42,123,2024,7,2025", help="Lista de seeds separadas por vírgula")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    models_base = Path("/media/ssd2/lucas.ocunha/models-tcc/experimental/reconstruction_recon_only")

    # 1. Carregar Datasets
    celeb_csv = Path("data/celeb_df/test.csv")
    celeb_crops = Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")
    df_celeb = pd.read_csv(celeb_csv)
    celeb_dataset = CelebDFDataset(df_celeb, celeb_crops, image_size=224)
    celeb_loader = DataLoader(
        celeb_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )

    df40_csv = Path("data/df40/test.csv")
    df_df40 = pd.read_csv(df40_csv)
    df40_dataset = DF40Dataset(df_df40, image_size=224)
    df40_loader = DataLoader(
        df40_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )

    logger.info(f"Iniciando Avaliação Cross-Dataset (Reconstruction recon_only) em {device}")
    logger.info(f"Celeb-DF v2: {len(df_celeb)} frames | DF-40: {len(df_df40)} frames")

    celeb_results = []
    df40_results = []

    for seed in seeds:
        seed_dir = models_base / f"recon_only_seed_{seed}"
        if not (seed_dir / "best.pt").exists():
            logger.warning(f"Checkpoint para Seed {seed} não encontrado em {seed_dir}. Pulando...")
            continue

        # Ler threshold calibrado
        cal_path = seed_dir / "calibration.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal_data = json.load(f)
            threshold = float(cal_data.get("frame_threshold", 0.5))
        else:
            threshold = 0.5

        logger.info(f"\n=======================================================")
        logger.info(f"▶️  Avaliando Seed {seed} (Threshold Youden = {threshold:.4f})...")
        logger.info(f"=======================================================")

        model = load_model(seed_dir, device=device)

        # Avaliar Celeb-DF v2
        c_res = evaluate_celeb_df(
            model=model,
            celeb_loader=celeb_loader,
            df_meta=df_celeb,
            threshold=threshold,
            results_dir=seed_dir / "results",
            device=device,
            seed=seed,
        )
        celeb_results.append(c_res)

        # Avaliar DF-40
        d_res = evaluate_df40(
            model=model,
            df40_loader=df40_loader,
            manifest_df=df_df40,
            threshold=threshold,
            results_dir=seed_dir / "results",
            device=device,
            seed=seed,
        )
        df40_results.append(d_res)

        del model
        torch.cuda.empty_cache()

    if not celeb_results:
        logger.error("Nenhuma seed avaliada.")
        return

    # 2. Avaliação do Ensemble de 5 Seeds
    logger.info("\n=======================================================")
    logger.info("🏆  Calculando Deep Ensemble das 5 Seeds de recon_only...")
    logger.info("=======================================================")

    ensemble_dir = models_base / "recon_only_5seeds" / "results"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Celeb Ensemble
    celeb_p_mean = np.mean([r["p_fakes"] for r in celeb_results], axis=0)
    celeb_y = celeb_results[0]["y_trues"]
    ens_frame_auc = safe_auc(celeb_y, celeb_p_mean)
    ens_thresh = np.mean([float(json.load(open(models_base / f"recon_only_seed_{s}" / "calibration.json")).get("frame_threshold", 0.5)) for s in seeds])

    df_temp = df_celeb.copy()
    df_temp["prob_fake"] = celeb_p_mean
    video_df = df_temp.groupby("video_id").agg({"target": "first", "prob_fake": "mean"}).reset_index()
    vid_y = video_df["target"].to_numpy().astype(int)
    vid_p = video_df["prob_fake"].to_numpy()
    ens_video_auc = safe_auc(vid_y, vid_p)
    ens_video_acc = float(accuracy_score(vid_y, (vid_p >= ens_thresh).astype(int)))

    np.savez_compressed(
        ensemble_dir / "outputs_celeb_df.npz",
        y_true=celeb_y, p_fake=celeb_p_mean, video_y=vid_y, video_p=vid_p, threshold=ens_thresh
    )

    # DF40 Ensemble
    df40_p_mean = np.mean([r["p_fakes"] for r in df40_results], axis=0)
    df40_y = df40_results[0]["y_trues"]
    ens_df40_auc = safe_auc(df40_y, df40_p_mean)
    ens_df40_acc = float(accuracy_score(df40_y, (df40_p_mean >= ens_thresh).astype(int)))
    ens_df40_f1 = float(f1_score(df40_y, (df40_p_mean >= ens_thresh).astype(int), zero_division=0))

    np.savez_compressed(
        ensemble_dir / "outputs_df40.npz",
        probs=df40_p_mean, y_true=df40_y, y_pred=(df40_p_mean >= ens_thresh).astype(int),
        threshold=ens_thresh, ids=df_df40.index.values
    )

    # 3. Relatório Final Consolidado
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL DA AVALIAÇÃO CROSS-DATASET: RESNET-18 RECON_ONLY")
    print("=" * 80)

    rows = []
    for c, d in zip(celeb_results, df40_results):
        rows.append({
            "Semente": f"Seed {c['seed']}",
            "Celeb Frame AUC": f"{c['frame_auc']*100:.2f}%",
            "Celeb Video AUC": f"{c['video_auc']*100:.2f}%",
            "Celeb Video Acc": f"{c['video_acc']*100:.2f}%",
            "DF-40 AUC": f"{d['auc']*100:.2f}%",
            "DF-40 Acc": f"{d['acc']*100:.2f}%",
            "DF-40 F1": f"{d['f1']*100:.2f}%",
        })

    # Média e std
    c_f_aucs = [r["frame_auc"] * 100 for r in celeb_results]
    c_v_aucs = [r["video_auc"] * 100 for r in celeb_results]
    c_v_accs = [r["video_acc"] * 100 for r in celeb_results]
    d_aucs = [r["auc"] * 100 for r in df40_results]
    d_accs = [r["acc"] * 100 for r in df40_results]
    d_f1s = [r["f1"] * 100 for r in df40_results]

    rows.append({
        "Semente": "MÉDIA ± STD",
        "Celeb Frame AUC": f"{np.mean(c_f_aucs):.2f}% ± {np.std(c_f_aucs):.2f}%",
        "Celeb Video AUC": f"{np.mean(c_v_aucs):.2f}% ± {np.std(c_v_aucs):.2f}%",
        "Celeb Video Acc": f"{np.mean(c_v_accs):.2f}% ± {np.std(c_v_accs):.2f}%",
        "DF-40 AUC": f"{np.mean(d_aucs):.2f}% ± {np.std(d_aucs):.2f}%",
        "DF-40 Acc": f"{np.mean(d_accs):.2f}% ± {np.std(d_accs):.2f}%",
        "DF-40 F1": f"{np.mean(d_f1s):.2f}% ± {np.std(d_f1s):.2f}%",
    })

    rows.append({
        "Semente": "DEEP ENSEMBLE (5 SEEDS)",
        "Celeb Frame AUC": f"{ens_frame_auc*100:.2f}%",
        "Celeb Video AUC": f"{ens_video_auc*100:.2f}%",
        "Celeb Video Acc": f"{ens_video_acc*100:.2f}%",
        "DF-40 AUC": f"{ens_df40_auc*100:.2f}%",
        "DF-40 Acc": f"{ens_df40_acc*100:.2f}%",
        "DF-40 F1": f"{ens_df40_f1*100:.2f}%",
    })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # Breakdown por Paradigma no DF-40
    print("\n--- BREAKDOWN POR PARADIGMA NO DF-40 (MÉDIA 5 SEEDS) ---")
    paradigms = sorted(df40_results[0]["paradigms"].keys())
    p_rows = []
    for p in paradigms:
        p_aucs = [r["paradigms"][p]["auc"] * 100 for r in df40_results]
        p_recs = [r["paradigms"][p]["recall"] * 100 for r in df40_results]
        p_rows.append({
            "Paradigma": p,
            "AUC Média": f"{np.mean(p_aucs):.2f}% ± {np.std(p_aucs):.2f}%",
            "Recall Médio": f"{np.mean(p_recs):.2f}% ± {np.std(p_recs):.2f}%",
        })
    print(pd.DataFrame(p_rows).to_string(index=False))

    # Salvar resumo consolidado
    summary_df.to_csv(models_base / "cross_dataset_summary.csv", index=False)
    logger.info(f"Resumo consolidado salvo em {models_base / 'cross_dataset_summary.csv'}")


if __name__ == "__main__":
    main()
