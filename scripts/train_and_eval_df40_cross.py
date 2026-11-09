"""Treina arquiteturas com DF-40 e avalia benchmark reverso (Test, Test-D, DF-40, Celeb-DF v2)."""

from __future__ import annotations

import os
os.environ["TCC_SKIP_UNREADABLE"] = "1"
import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.data.data import ImageDataset
from src.models.registry import get_model_spec
from src.pipelines.config import load_config
from src.pipelines.evaluation import amp_context, best_threshold, sanitize_inputs, sanitize_logits

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("df40_cross")


def get_balanced_sampler(dataset: ImageDataset, seed: int = 42) -> WeightedRandomSampler:
    labels = dataset.df.iloc[:, 1].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    class_weights = np.divide(1.0, counts, out=np.zeros(2, dtype=float), where=counts > 0)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def eval_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_probs = []

    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, targets = batch[0], batch[1]
            images = sanitize_inputs(images.to(device, non_blocking=True))
            with amp_context(device):
                logits = model(images)
            logits = sanitize_logits(logits)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.tolist())

            if (i + 1) % 200 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - t0
                speed = len(all_targets) / max(elapsed, 1e-4)
                print(f"[{desc}] {len(all_targets)}/{len(loader.dataset)} ({speed:.1f} imgs/s)", flush=True)

    return np.array(all_targets, dtype=int), np.array(all_probs, dtype=float)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    if len(np.unique(y_true)) < 2:
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true, probs))
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    return {"auc": auc, "acc": acc, "f1": f1, "threshold": threshold, "n_samples": len(y_true)}


def evaluate_celeb_df(
    model: nn.Module,
    loader: DataLoader,
    manifest_csv: Path,
    device: torch.device,
    threshold: float,
) -> dict:
    y_trues, p_fakes = eval_dataset(model, loader, device, desc="Celeb-DF v2")
    frame_auc = float(roc_auc_score(y_trues, p_fakes))
    frame_preds = (p_fakes >= threshold).astype(int)
    frame_acc = float(accuracy_score(y_trues, frame_preds))
    frame_f1 = float(f1_score(y_trues, frame_preds, zero_division=0))

    df_meta = pd.read_csv(manifest_csv)
    df_meta["prob_fake"] = p_fakes
    video_df = df_meta.groupby("video_id").agg({"target": "first", "prob_fake": "mean"}).reset_index()

    vid_y = video_df["target"].to_numpy().astype(int)
    vid_p = video_df["prob_fake"].to_numpy()
    vid_preds = (vid_p >= threshold).astype(int)

    video_auc = float(roc_auc_score(vid_y, vid_p))
    video_acc = float(accuracy_score(vid_y, vid_preds))
    video_f1 = float(f1_score(vid_y, vid_preds, zero_division=0))

    return {
        "frame_auc": frame_auc, "frame_acc": frame_acc, "frame_f1": frame_f1,
        "video_auc": video_auc, "video_acc": video_acc, "video_f1": video_f1,
        "threshold": threshold, "n_frames": len(y_trues), "n_videos": len(vid_y)
    }


def main():
    parser = argparse.ArgumentParser(description="Treina modelo no DF-40 e avalia benchmark cruzado reverso")
    parser.add_argument("--family", default="dino", choices=["dino", "resnet", "mobilenet", "xception", "vit", "clip"])
    parser.add_argument("--mode", default="srm", choices=["none", "srm", "srm_only", "concat_frequency"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-only", action="store_true", help="Pular treino e avaliar checkpoint existente")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==============================================================================")
    print(f"🚀 TREINAMENTO DF-40 E BENCHMARK REVERSO CROSS-DATASET")
    print(f"Modelo: {args.family} | Modo: {args.mode} | Dispositivo: {device}")
    print(f"==============================================================================")

    # 1. Configuração do Modelo
    config_file = Path("configs") / f"{args.family}.yaml"
    cfg = load_config(config_file, {
        "fourier_mode": args.mode,
        "regime": "finetune",
        "seed": args.seed,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "batch_size": args.batch_size,
    })

    spec = get_model_spec(args.family)
    model = spec.build(cfg).to(device)
    if hasattr(spec, "unfreeze_for_finetune"):
        spec.unfreeze_for_finetune(model, cfg.unfreeze_last_n)

    output_dir = Path("models") / "df40_trained" / f"{args.family}_{args.mode}_seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / "best.pth"

    # Transformação padrão
    base_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_csv = Path("data/df40/train.csv")
    val_csv = Path("data/df40/val.csv")
    df40_test_csv = Path("data/df40/test.csv")
    mffi_test_csv = Path("data/raw/test.csv")
    celeb_csv = Path("data/celeb_df/test.csv")

    assert train_csv.exists() and val_csv.exists(), "Splits data/df40/train.csv e val.csv não encontrados!"

    # 2. Treinamento
    if not args.eval_only:
        train_ds = ImageDataset(
            file_csv=train_csv, images_dir=Path(""), transform=base_tf,
            fourier=args.mode, spatial_size=(cfg.image_size, cfg.image_size), in_channels=cfg.in_channels,
        )
        val_ds = ImageDataset(
            file_csv=val_csv, images_dir=Path(""), transform=base_tf,
            fourier=args.mode, spatial_size=(cfg.image_size, cfg.image_size), in_channels=cfg.in_channels,
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=get_balanced_sampler(train_ds, args.seed),
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        criterion = nn.CrossEntropyLoss()
        head_params = [p for p in model.classifier.parameters() if p.requires_grad] if hasattr(model, "classifier") else []
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad] if hasattr(model, "backbone") else []
        if head_params or bb_params:
            param_groups = [
                {"params": head_params, "lr": args.lr_head},
                {"params": bb_params, "lr": args.lr_backbone},
            ]
        else:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_head}]
        optimizer = AdamW(param_groups, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_val_score = -1.0
        best_threshold_val = 0.5
        history = []

        print(f"\nIniciando treinamento ({args.epochs} épocas)...")
        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            t_epoch = time.time()

            for i, batch in enumerate(train_loader):
                images, targets = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()
                with amp_context(device):
                    logits = model(images)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_losses.append(loss.item())

            # Validação
            val_y, val_p = eval_dataset(model, val_loader, device, desc=f"Val E{epoch+1}")
            val_thresh, _ = best_threshold(val_y, val_p, metric="f1")
            val_metrics = compute_metrics(val_y, val_p, threshold=val_thresh)

            epoch_time = time.time() - t_epoch
            train_loss_avg = np.mean(train_losses)
            val_score = val_metrics["f1"] + val_metrics["auc"]

            print(
                f"Época {epoch+1:02d}/{args.epochs:02d} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss_avg:.4f} | "
                f"Val AUC: {val_metrics['auc']*100:.2f}% | "
                f"Val Acc: {val_metrics['acc']*100:.2f}% | "
                f"Val F1: {val_metrics['f1']*100:.2f}% (thresh={val_thresh:.4f})",
                flush=True,
            )

            scheduler.step(val_score)

            if val_score > best_val_score:
                best_val_score = val_score
                best_threshold_val = val_thresh
                torch.save(model.state_dict(), best_ckpt)
                print(f" ⭐ Melhor checkpoint salvo em {best_ckpt} (score={val_score:.4f})", flush=True)

            history.append({
                "epoch": epoch + 1, "train_loss": train_loss_avg,
                "val_auc": val_metrics["auc"], "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"], "threshold": val_thresh,
            })

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    else:
        history_csv = output_dir / "history.csv"
        if history_csv.exists():
            h_df = pd.read_csv(history_csv)
            best_idx = (h_df["val_auc"] + h_df["val_f1"]).idxmax()
            best_threshold_val = float(h_df.loc[best_idx, "threshold"])
            print(f"Lido threshold de validação ótimo do history.csv: {best_threshold_val:.4f}")
        else:
            best_threshold_val = 0.5

    # 3. Carregar Melhor Modelo para Avaliação nos 4 Alvos
    print(f"\n==============================================================================")
    print(f"📊 AVALIANDO MELHOR MODELO TREINADO NO DF-40 EM TODOS OS 4 ALVOS")
    print(f"Checkpoint: {best_ckpt}")
    print(f"==============================================================================")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    # ALVO 1: DF-40 Test (Intra-dataset)
    print("\n[1/4] Avaliando DF-40 Test (Intra-dataset)...")
    df40_test_ds = ImageDataset(
        file_csv=df40_test_csv, images_dir=Path(""), transform=base_tf,
        fourier=args.mode, spatial_size=(cfg.image_size, cfg.image_size), in_channels=cfg.in_channels,
    )
    df40_loader = DataLoader(df40_test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    df40_y, df40_p = eval_dataset(model, df40_loader, device, desc="DF-40 Test")
    df40_res = compute_metrics(df40_y, df40_p, threshold=best_threshold_val)
    print(f" -> DF-40 Test: AUC = {df40_res['auc']*100:.2f}% | Acc = {df40_res['acc']*100:.2f}%")

    # ALVO 2: Celeb-DF v2 (Cross-dataset vídeo)
    print("\n[2/4] Avaliando Celeb-DF v2 (Cross-dataset)...")
    celeb_ds = ImageDataset(
        file_csv=celeb_csv, images_dir=Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops"), transform=base_tf,
        fourier=args.mode, spatial_size=(cfg.image_size, cfg.image_size), in_channels=cfg.in_channels,
    )
    celeb_loader = DataLoader(celeb_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    celeb_res = evaluate_celeb_df(model, celeb_loader, celeb_csv, device, threshold=best_threshold_val)
    print(f" -> Celeb-DF v2: Video AUC = {celeb_res['video_auc']*100:.2f}% | Video Acc = {celeb_res['video_acc']*100:.2f}%")
    print(f" -> Celeb-DF v2: Frame AUC = {celeb_res['frame_auc']*100:.2f}% | Frame Acc = {celeb_res['frame_acc']*100:.2f}%")

    # ALVO 3: MFFI Test Limpo (Cross-dataset)
    print("\n[3/4] Avaliando MFFI Test Limpo (Cross-dataset)...")
    mffi_clean_ds = ImageDataset(
        file_csv=mffi_test_csv, images_dir=Path("/media/ssd2/lucas.ocunha/datasets/phase1/testset"), transform=base_tf,
        fourier=args.mode, spatial_size=(cfg.image_size, cfg.image_size), in_channels=cfg.in_channels,
    )
    mffi_clean_loader = DataLoader(mffi_clean_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    mffi_y, mffi_p = eval_dataset(model, mffi_clean_loader, device, desc="MFFI Test Limpo")
    mffi_res = compute_metrics(mffi_y, mffi_p, threshold=best_threshold_val)
    print(f" -> MFFI Test: AUC = {mffi_res['auc']*100:.2f}% | Acc = {mffi_res['acc']*100:.2f}%")

    # ALVO 4: MFFI Test-D Corrompido (Cross-dataset sob corrupções)
    print("\n[4/4] Avaliando MFFI Test-D Corrompido (Cross-dataset)...")
    mffi_testd_ds = ImageDataset(
        file_csv=mffi_test_csv, images_dir=Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d"), transform=base_tf,
        fourier=args.mode, spatial_size=(cfg.image_size, cfg.image_size), in_channels=cfg.in_channels,
    )
    mffi_testd_loader = DataLoader(mffi_testd_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    testd_y, testd_p = eval_dataset(model, mffi_testd_loader, device, desc="MFFI Test-D")
    testd_res = compute_metrics(testd_y, testd_p, threshold=best_threshold_val)
    print(f" -> MFFI Test-D: AUC = {testd_res['auc']*100:.2f}% | Acc = {testd_res['acc']*100:.2f}%")

    # Consolidar Resultados
    results_dict = {
        "model_family": args.family,
        "fourier_mode": args.mode,
        "train_dataset": "DF-40",
        "threshold": best_threshold_val,
        "df40_test": df40_res,
        "celeb_df": celeb_res,
        "mffi_test": mffi_res,
        "mffi_test_d": testd_res,
    }

    results_file = output_dir / "cross_dataset_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Resumo em Tabela
    print(f"\n==============================================================================")
    print(f"🏁 RESULTADOS FINAIS: MODELO {args.family.upper()} ({args.mode.upper()}) TREINADO NO DF-40")
    print(f"==============================================================================")
    print(f"| Alvo (Target)               | AUC (%)  | Acc (%)  | Tipo de Avaliação |")
    print(f"|-----------------------------|----------|----------|-------------------|")
    print(f"| DF-40 Test                  | {df40_res['auc']*100:6.2f}%  | {df40_res['acc']*100:6.2f}%  | Intra-dataset     |")
    print(f"| Celeb-DF v2 (Vídeo)         | {celeb_res['video_auc']*100:6.2f}%  | {celeb_res['video_acc']*100:6.2f}%  | Cross-dataset     |")
    print(f"| Celeb-DF v2 (Frame)         | {celeb_res['frame_auc']*100:6.2f}%  | {celeb_res['frame_acc']*100:6.2f}%  | Cross-dataset     |")
    print(f"| MFFI Test (Limpo)           | {mffi_res['auc']*100:6.2f}%  | {mffi_res['acc']*100:6.2f}%  | Cross-dataset     |")
    print(f"| MFFI Test-D (Corrompido)    | {testd_res['auc']*100:6.2f}%  | {testd_res['acc']*100:6.2f}%  | Cross-dataset     |")
    print(f"==============================================================================")
    print(f"Resultados gravados em: {results_file}")


if __name__ == "__main__":
    main()
