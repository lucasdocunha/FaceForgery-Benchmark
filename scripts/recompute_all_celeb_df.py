#!/usr/bin/env python3
"""Recomputa a avaliação do Celeb-DF v2 (Frame e Vídeo) em todos os checkpoints existentes.

Corrige o problema de métricas antigas/invertidas, garantindo a semântica correta de classes:
- target = 1: DeepFake / Manipulado (P(fake) = softmax[:, 1])
- target = 0: Real
Salva metrics_celeb_df.csv, metrics_celeb_df_video.csv e outputs_celeb_df.npz no padrão do repositório.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.augmentations import clean_transform
from src.data.data import ImageDataset
from src.data.paths import data_root, models_root
from src.models.registry import get_model_spec
from src.pipelines.checkpoints import discover_trained_runs
from src.pipelines.config import load_config


def evaluate_single_run(
    run,
    celeb_loader: DataLoader,
    df_meta: pd.DataFrame,
    device: torch.device,
    force: bool = False,
) -> dict | None:
    res_dir = run.run_dir / "results"
    v_csv = res_dir / "metrics_celeb_df_video.csv"
    f_csv = res_dir / "metrics_celeb_df.csv"
    npz_path = res_dir / "outputs_celeb_df.npz"

    # Critério de já reavaliado: se outputs_celeb_df.npz existe com chave video_p e video_auc > 0.45
    if not force and v_csv.exists() and f_csv.exists() and npz_path.exists():
        try:
            df_v = pd.read_csv(v_csv)
            if not df_v.empty and "auc" in df_v.columns:
                v_auc = float(df_v.iloc[0]["auc"])
                # Se já foi reavaliado recentemente e tem formato novo
                data_npz = np.load(npz_path)
                if "video_p" in data_npz and v_auc > 0.45:
                    return {
                        "family": run.model_family, "mode": run.fourier_mode,
                        "regime": run.regime, "seed": run.seed,
                        "video_auc": v_auc, "cached": True,
                    }
        except Exception:
            pass

    # Carrega threshold da validação (se disponível)
    val_csv = res_dir / "metrics_val.csv"
    val_threshold = 0.5
    if val_csv.exists():
        try:
            df_val = pd.read_csv(val_csv)
            if not df_val.empty and "threshold" in df_val.columns:
                val_threshold = float(df_val.iloc[0]["threshold"])
        except Exception:
            val_threshold = 0.5

    # Constrói modelo
    config_path = ROOT_DIR / "configs" / f"{run.model_family}.yaml"
    overrides = {
        "model_family": run.model_family,
        "fourier_mode": run.fourier_mode,
        "regime": "finetune" if "finetune" in run.regime else "scratch",
        "seed": run.seed,
    }
    config = load_config(config_path, overrides)
    spec = get_model_spec(run.model_family)
    model = spec.build(config)

    # Carrega pesos
    try:
        state = torch.load(run.weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except Exception as e:
        print(f"❌ Erro ao carregar pesos de {run.weights_path}: {e}")
        return None

    model = model.to(device).eval()

    y_trues, p_fakes = [], []
    with torch.no_grad():
        for batch in celeb_loader:
            x = batch[0].to(device)
            y = batch[1]
            logits = model(x)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            y_trues.extend(y.numpy().tolist())
            p_fakes.extend(probs.tolist())

    y_trues = np.array(y_trues)
    p_fakes = np.nan_to_num(np.array(p_fakes), nan=0.5, posinf=1.0, neginf=0.0)

    # Métricas de Frame
    try:
        frame_auc = float(roc_auc_score(y_trues, p_fakes))
    except Exception:
        frame_auc = 0.5
    frame_preds = (p_fakes >= val_threshold).astype(int)
    frame_acc = float(accuracy_score(y_trues, frame_preds))
    frame_f1 = float(f1_score(y_trues, frame_preds, zero_division=0))
    frame_p = float(precision_score(y_trues, frame_preds, zero_division=0))
    frame_r = float(recall_score(y_trues, frame_preds, zero_division=0))
    frame_spec = float(recall_score(1 - y_trues, 1 - frame_preds, zero_division=0))

    # Métricas de Vídeo (agregação por média de probabilidade por vídeo)
    df_temp = df_meta.copy()
    df_temp["prob_fake"] = p_fakes
    video_df = df_temp.groupby("video_id").agg({"target": "first", "prob_fake": "mean"}).reset_index()

    vid_y = video_df["target"].to_numpy().astype(int)
    vid_p = np.nan_to_num(video_df["prob_fake"].to_numpy(), nan=0.5, posinf=1.0, neginf=0.0)
    vid_preds = (vid_p >= val_threshold).astype(int)

    try:
        video_auc = float(roc_auc_score(vid_y, vid_p))
    except Exception:
        video_auc = 0.5
    video_acc = float(accuracy_score(vid_y, vid_preds))
    video_f1 = float(f1_score(vid_y, vid_preds, zero_division=0))
    video_p = float(precision_score(vid_y, vid_preds, zero_division=0))
    video_r = float(recall_score(vid_y, vid_preds, zero_division=0))
    video_spec = float(recall_score(1 - vid_y, 1 - vid_preds, zero_division=0))

    # Salva arquivos atualizados
    res_dir.mkdir(parents=True, exist_ok=True)

    row_frame = {
        "model_family": run.model_family, "fourier_mode": run.fourier_mode,
        "regime": run.regime, "seed": run.seed, "split": "celeb_df",
        "threshold": val_threshold, "auc": frame_auc, "acc": frame_acc,
        "f1": frame_f1, "precision": frame_p, "recall": frame_r,
        "specificity": frame_spec, "loss": 0.0,
        "frame_auc": frame_auc, "frame_acc": frame_acc, "frame_f1": frame_f1,
        "video_auc": video_auc, "video_acc": video_acc, "video_f1": video_f1,
    }
    pd.DataFrame([row_frame]).to_csv(f_csv, index=False)

    row_video = {
        "model_family": run.model_family, "fourier_mode": run.fourier_mode,
        "regime": run.regime, "seed": run.seed, "split": "celeb_df_video",
        "threshold": val_threshold, "auc": video_auc, "acc": video_acc,
        "f1": video_f1, "precision": video_p, "recall": video_r,
        "specificity": video_spec, "loss": 0.0,
    }
    pd.DataFrame([row_video]).to_csv(v_csv, index=False)

    np.savez_compressed(npz_path, y_true=y_trues, p_fake=p_fakes, video_y=vid_y, video_p=vid_p)

    return {
        "family": run.model_family, "mode": run.fourier_mode,
        "regime": run.regime, "seed": run.seed,
        "frame_auc": frame_auc, "video_auc": video_auc,
        "cached": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reavalia Celeb-DF v2 em todos os checkpoints com o código corrigido")
    parser.add_argument("--gpu", type=int, default=0, help="ID da GPU (0 ou 1)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size de inferência")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers do DataLoader")
    parser.add_argument("--families", type=str, default=None, help="Filtrar famílias separadas por vírgula")
    parser.add_argument("--modes", type=str, default=None, help="Filtrar modos separados por vírgula")
    parser.add_argument("--seeds", type=str, default=None, help="Filtrar sementes separadas por vírgula")
    parser.add_argument("--force", action="store_true", help="Forçar reavaliação de todos os modelos")
    parser.add_argument("--dry-run", action="store_true", help="Apenas listar os modelos a processar")
    args = parser.parse_args(argv)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Dispositivo selecionado: {device}")

    celeb_csv = data_root() / "celeb_df" / "test.csv"
    crops_dir = Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")

    if not celeb_csv.exists() or not crops_dir.exists():
        print(f"❌ Celeb-DF não encontrado em {celeb_csv} ou {crops_dir}!")
        sys.exit(1)

    df_meta = pd.read_csv(celeb_csv)
    print(f"📁 Metadados Celeb-DF: {len(df_meta)} frames de {df_meta['video_id'].nunique()} vídeos.")

    root = models_root()
    runs = discover_trained_runs(root)
    print(f"🔍 Total de checkpoints descobertos: {len(runs)}")

    # Filtros
    if args.families:
        fam_set = set(f.strip() for f in args.families.split(","))
        runs = [r for r in runs if r.model_family in fam_set]
    if args.modes:
        mode_set = set(m.strip() for m in args.modes.split(","))
        runs = [r for r in runs if r.fourier_mode in mode_set]
    if args.seeds:
        seed_set = set(f"seed_{s.strip()}" if not s.strip().startswith("seed_") else s.strip() for s in args.seeds.split(","))
        runs = [r for r in runs if r.seed in seed_set]

    # Ordena agrupando por (fourier_mode, image_size) para reaproveitar DataLoaders!
    runs = sorted(runs, key=lambda r: (r.fourier_mode, r.model_family, r.regime, r.seed))
    print(f"🎯 Total de runs filtrados para reavaliação: {len(runs)}")

    if args.dry_run:
        for r in runs:
            print(f"  • {r.model_family:12s} | {r.fourier_mode:16s} | {r.regime:16s} | {r.seed}")
        return

    # Cache de DataLoaders por (fourier_mode, image_size)
    loader_cache = {}

    t0 = time.time()
    recomputed = 0
    cached_count = 0

    for idx, r in enumerate(runs, 1):
        # Determina image_size
        config_path = ROOT_DIR / "configs" / f"{r.model_family}.yaml"
        cfg = load_config(config_path, {"model_family": r.model_family})
        img_size = cfg.image_size

        cache_key = (r.fourier_mode, img_size)
        if cache_key not in loader_cache:
            print(f"\n📦 Criando DataLoader para (modo={r.fourier_mode}, size={img_size})...", flush=True)
            ds = ImageDataset(
                celeb_csv, crops_dir, transform=clean_transform(img_size),
                fourier=r.fourier_mode, spatial_size=(img_size, img_size),
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
            )
            loader_cache[cache_key] = loader
        else:
            loader = loader_cache[cache_key]

        t_run = time.time()
        res = evaluate_single_run(r, loader, df_meta, device, force=args.force)
        dt = time.time() - t_run

        if res is None:
            continue

        if res.get("cached", False):
            cached_count += 1
            print(f"[{idx:3d}/{len(runs)}] ⏭️  {r.model_family:10s} | {r.fourier_mode:14s} | {r.regime:16s} | {r.seed} (Já corrigido: Video AUC {res['video_auc']*100:.2f}%)", flush=True)
        else:
            recomputed += 1
            print(f"[{idx:3d}/{len(runs)}] ✅ {r.model_family:10s} | {r.fourier_mode:14s} | {r.regime:16s} | {r.seed} -> Frame AUC: {res['frame_auc']*100:.2f}% | Video AUC: {res['video_auc']*100:.2f}% ({dt:.1f}s)", flush=True)

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\n🎉 CONCLUÍDO! Reavaliados: {recomputed} | Já em cache: {cached_count} | Tempo total: {elapsed_min:.1f} min")


if __name__ == "__main__":
    main()
