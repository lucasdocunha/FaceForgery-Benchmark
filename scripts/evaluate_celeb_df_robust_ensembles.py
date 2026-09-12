"""Avaliação dos 6 modelos robustos (Seed 987) e seus ensembles no Celeb-DF v2."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data import ImageDataset
from src.data.paths import models_root
from src.pipelines.checkpoints import discover_trained_runs, load_model_from_run, config_from_run
from src.pipelines.evaluation import binary_metrics, evaluate_classifier


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = models_root()
    manifest_csv = Path("data/celeb_df/test.csv")
    crops_dir = Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")

    models = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]

    predictions = {}
    manifest_df = pd.read_csv(manifest_csv)

    print("Gerando predições dos 6 modelos robustos no Celeb-DF v2...", flush=True)
    for m in models:
        runs = [
            r for r in discover_trained_runs(root, only_model_family=m)
            if r.seed == 987 and r.fourier_mode == "none" and r.regime == "finetune"
        ]
        if not runs:
            print(f"Run não encontrado para {m}", flush=True)
            continue
        run = runs[0]
        config = config_from_run(run)
        model = load_model_from_run(run, device)

        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        dataset = ImageDataset(
            manifest_csv,
            crops_dir,
            transform=transform,
            fourier="none",
            spatial_size=(config.image_size, config.image_size),
            in_channels=3,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        res = evaluate_classifier(
            model, loader, torch.nn.CrossEntropyLoss(), device,
            threshold=run.threshold, use_amp=True, desc=m
        )
        predictions[m] = res["probs"]
        del model
        torch.cuda.empty_cache()

    y_true = manifest_df["target"].values
    video_ids = manifest_df["video_id"].values

    def eval_preds(probs_dict, model_keys, fusion_func):
        stacked = np.stack([probs_dict[k] for k in model_keys], axis=0)
        if fusion_func == "mean":
            fused = np.mean(stacked, axis=0)
        elif fusion_func == "geom":
            clipped = np.clip(stacked, 1e-7, 1 - 1e-7)
            log_mean = np.mean(np.log(clipped), axis=0)
            fused = np.exp(log_mean)
        elif fusion_func == "max":
            fused = np.max(stacked, axis=0)
        elif fusion_func == "min":
            fused = np.min(stacked, axis=0)
        else:
            raise ValueError(fusion_func)

        frame_m = binary_metrics(y_true, fused, threshold=0.5)

        df = pd.DataFrame({"video_id": video_ids, "prob": fused, "target": y_true})
        v_group = df.groupby("video_id")
        v_prob = v_group["prob"].mean().values
        v_true = v_group["target"].first().values
        video_m = binary_metrics(v_true, v_prob, threshold=0.5)

        return frame_m, video_m

    combinations = [
        ("CLIP + DINO", ["clip", "dino"]),
        ("CLIP + DINO + XCEPTION", ["clip", "dino", "xception"]),
        ("CLIP + DINO + RESNET", ["clip", "dino", "resnet"]),
        ("CLIP + DINO + VIT", ["clip", "dino", "vit"]),
        ("DINO + XCEPTION + VIT", ["dino", "xception", "vit"]),
        ("TODOS OS 6 ROBUSTOS", models),
    ]

    ensemble_rows = []
    for name, m_keys in combinations:
        for strategy in ["mean", "geom", "max"]:
            fm, vm = eval_preds(predictions, m_keys, strategy)
            ensemble_rows.append({
                "Ensemble": name,
                "Fusão": strategy,
                "Frame AUC (%)": round(fm["auc"] * 100, 2),
                "Frame ACC (%)": round(fm["acc"] * 100, 2),
                "Video AUC (%)": round(vm["auc"] * 100, 2),
                "Video ACC (%)": round(vm["acc"] * 100, 2),
                "Video F1 (%)": round(vm["f1"] * 100, 2),
            })

    ens_df = pd.DataFrame(ensemble_rows)
    print("\n=== RESULTADOS DOS ENSEMBLES ROBUSTOS NO CELEB-DF v2 ===", flush=True)
    sorted_df = ens_df.sort_values(by="Video AUC (%)", ascending=False)
    print(sorted_df.to_markdown(index=False), flush=True)

    out_csv = Path("tables/celeb_df_robust_ensembles.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sorted_df.to_csv(out_csv, index=False)
    print(f"Salvo em {out_csv}", flush=True)


if __name__ == "__main__":
    main()
