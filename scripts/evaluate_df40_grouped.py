#!/usr/bin/env python3
"""
Inferência Exaustiva Agrupada no DF-40 (650.549 Imagens Fakes).
Processa MÚLTIPLOS modelos que compartilham a mesma representação de entrada
em UMA ÚNICA PASSADA pelo dataset, eliminando leituras e cálculos de FFT redundantes.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.forensics.srm import extract_srm_residuals
from src.models.registry import get_model_spec
from src.pipelines.config import load_config
from scripts.evaluate_df40_all_fakes_generic import GenericDF40Dataset, collect_images_for_folder


def main():
    parser = argparse.ArgumentParser(description="Inferência Agrupada DF-40")
    parser.add_argument("--mode", type=str, required=True, help="Modo Fourier/Representação (none, srm, srm_only, magnitude, phase, complex, frequency_3, concat, concat_frequency)")
    parser.add_argument("--models-json", type=str, required=True, help="Lista de modelos em JSON (ou caminho para arquivo .json)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo PyTorch (cuda:0, cuda:1)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size do DataLoader")
    parser.add_argument("--num-workers", type=int, default=10, help="Workers DataLoader")
    parser.add_argument("--output-dir", type=str, default="tables/df40_all_fakes_models", help="Diretório de saída")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Carregar lista de modelos
    if os.path.exists(args.models_json):
        with open(args.models_json) as f:
            models_defs = json.load(f)
    else:
        models_defs = json.loads(args.models_json)

    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filtrar modelos que já foram concluídos
    active_models_defs = []
    for m in models_defs:
        out_csv = out_dir / f"df40_all_fakes_{m['run_name']}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 100:
            print(f"⏩ [Já Concluído] Pulando {m['run_name']} ({out_csv.name} já existe)")
        else:
            active_models_defs.append(m)

    if not active_models_defs:
        print("✅ Todos os modelos deste grupo já foram concluídos. Nada a fazer!")
        return

    print("=" * 88)
    print(f"🚀 INFERÊNCIA AGRUPADA DF-40 | MODO: {args.mode.upper()} | DISPOSITIVO: {device}")
    print(f"Total de modelos simultâneos no lote: {len(active_models_defs)}")
    for m in active_models_defs:
        print(f"  • {m['run_name']} ({m['family']}) -> {m['checkpoint']}")
    print("=" * 88)

    # Carregar cada modelo na GPU
    loaded_models = []
    max_in_channels = 3

    for m in active_models_defs:
        ckpt_path = Path(m["checkpoint"])
        if not ckpt_path.exists():
            print(f"❌ Checkpoint não encontrado: {ckpt_path}")
            continue

        # Descobrir threshold
        threshold = m.get("threshold")
        if threshold is None:
            val_csv = ckpt_path.parent.parent / "results" / "metrics_val.csv"
            if val_csv.exists():
                try:
                    df_val = pd.read_csv(val_csv)
                    if "threshold" in df_val.columns:
                        threshold = float(df_val["threshold"].iloc[0])
                    elif "best_threshold" in df_val.columns:
                        threshold = float(df_val["best_threshold"].iloc[0])
                except Exception:
                    pass
        if threshold is None:
            threshold = 0.50

        # Carregar config e modelo
        config_file = ROOT_DIR / "configs" / f"{m['family']}.yaml"
        cfg = load_config(config_file, {"fourier_mode": args.mode, "regime": "finetune"})

        # In-channels override
        run_cfg_path = ckpt_path.parent.parent / "results" / "run_config.json"
        if run_cfg_path.exists():
            try:
                with open(run_cfg_path) as f:
                    rc = json.load(f)
                    if "in_channels" in rc:
                        cfg.in_channels_override = int(rc["in_channels"])
            except Exception:
                pass

        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        for k, v in state.items():
            if "weight" in k and v.ndim == 4 and v.shape[1] in (1, 2, 3, 4, 6, 7, 8):
                cfg.in_channels_override = v.shape[1]
                break

        in_ch = cfg.in_channels
        if in_ch > max_in_channels:
            max_in_channels = in_ch

        spec = get_model_spec(m["family"])
        net = spec.build(cfg).to(device).eval()
        net.load_state_dict(state)

        loaded_models.append({
            "run_name": m["run_name"],
            "family": m["family"],
            "model": net,
            "threshold": threshold,
            "in_channels": in_ch,
            "results": [],
            "total_detected_all": 0,
            "total_imgs_all": 0,
            "out_csv": out_dir / f"df40_all_fakes_{m['run_name']}.csv",
        })
        print(f"  ✅ Carregado: {m['run_name']} (in_channels={in_ch}, threshold={threshold:.4f})")

    if not loaded_models:
        print("❌ Nenhum modelo foi carregado com sucesso.")
        sys.exit(1)

    # Carregar cache de caminhos das 24 técnicas
    root = Path("/media/ssd2/lucas.ocunha/datasets/df40_extracted")
    cache_path = root / "paths_cache.json"
    paths_cache = None
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                paths_cache = json.load(f)
            print(f"⚡ [Cache de Imagens Ativo] {sum(len(v) for v in paths_cache.values()):,} caminhos carregados instantaneamente.")
        except Exception:
            paths_cache = None

    techniques = [
        # Difusão
        ("DiT", "DiT", "Difusão"),
        ("SiT", "SiT", "Difusão"),
        ("ddim", "ddim", "Difusão"),
        ("sd2.1", "sd2.1", "Difusão"),
        ("pixart", "pixart", "Difusão"),
        ("RDDM", "RDDM", "Difusão"),
        ("CollabDiff/fake", "CollabDiff", "Difusão"),
        # Síntese GAN
        ("StyleGAN2", "StyleGAN2", "Síntese GAN"),
        ("StyleGAN3", "StyleGAN3", "Síntese GAN"),
        ("StyleGANXL", "StyleGANXL", "Síntese GAN"),
        ("VQGAN", "VQGAN", "Síntese GAN"),
        ("stargan/fake", "stargan", "Síntese GAN"),
        ("starganv2/fake", "starganv2", "Síntese GAN"),
        # Troca Facial
        ("deepfacelab/fake", "deepfacelab", "Troca Facial"),
        ("faceswap", "faceswap", "Troca Facial"),
        ("mobileswap", "mobileswap", "Troca Facial"),
        ("blendface", "blendface", "Troca Facial"),
        ("uniface", "uniface", "Troca Facial"),
        # Edição / T2I
        ("MidJourney/fake", "MidJourney", "Edição / T2I"),
        ("whichfaceisreal/fake", "whichfaceisreal", "Edição / T2I"),
        ("styleclip/fake", "styleclip", "Edição / T2I"),
        ("e4e", "e4e", "Edição / T2I"),
        # Reencenação & Avatares
        ("heygen_new/fake", "heygen", "Avatares & Reencenação"),
        ("cdf/frames", "cdf_reenactment", "Avatares & Reencenação"),
    ]

    print("\n" + "=" * 88)
    print(f"🏁 INICIANDO LOOP DE TÉCNICAS (TOTAL: {len(techniques)} TÉCNICAS DF-40)")
    print("=" * 88)

    t_start_group = time.time()

    for rel_path, name, paradigm in techniques:
        folder = root / rel_path
        if not folder.exists():
            continue

        if paths_cache and name in paths_cache:
            imgs = [Path(p) for p in paths_cache[name]]
        else:
            imgs = collect_images_for_folder(folder)

        n_imgs = len(imgs)
        if n_imgs == 0:
            continue

        ds = GenericDF40Dataset(imgs, img_size=224, fourier_mode=args.mode, in_channels=max_in_channels)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        t0 = time.time()
        tech_stats = [{"thresh": 0, "05": 0, "sum_probs": 0.0} for _ in loaded_models]

        with torch.inference_mode():
            for x in loader:
                x = x.to(device, non_blocking=True)
                if args.mode == "srm":
                    res = extract_srm_residuals(x)
                    inp = torch.cat([x, res], dim=1)
                elif args.mode == "srm_only":
                    inp = extract_srm_residuals(x)
                else:
                    inp = x

                with torch.amp.autocast("cuda"):
                    for idx, lm in enumerate(loaded_models):
                        # Ajuste de canais se necessário
                        m_inp = inp[:, :lm["in_channels"]] if inp.shape[1] > lm["in_channels"] else inp
                        logits = lm["model"](m_inp)
                        probs = torch.softmax(logits.float(), dim=-1)[:, 1]

                        tech_stats[idx]["thresh"] += (probs >= lm["threshold"]).sum().item()
                        tech_stats[idx]["05"] += (probs >= 0.50).sum().item()
                        tech_stats[idx]["sum_probs"] += probs.sum().item()

        dt = time.time() - t0

        # Gravar estatísticas para cada modelo
        acc_strs = []
        for idx, lm in enumerate(loaded_models):
            n_det = tech_stats[idx]["thresh"]
            acc = (n_det / n_imgs) * 100
            acc_05 = (tech_stats[idx]["05"] / n_imgs) * 100
            mean_prob = tech_stats[idx]["sum_probs"] / n_imgs

            lm["total_imgs_all"] += n_imgs
            lm["total_detected_all"] += n_det
            lm["results"].append({
                "model_name": lm["run_name"],
                "family": lm["family"],
                "mode": args.mode,
                "paradigma": paradigm,
                "tecnica": name,
                "n_amostras_total": n_imgs,
                "acertos_threshold_val": n_det,
                "taxa_acerto_pct": round(acc, 2),
                "taxa_acerto_05_pct": round(acc_05, 2),
                "prob_fake_media": round(mean_prob, 4),
                "tempo_s": round(dt, 1),
            })
            acc_strs.append(f"{lm['family']}: {acc:5.1f}%")

        print(f"[{time.strftime('%X')}] {paradigm:20s} | {name:16s} ({n_imgs:6d} imgs, {dt:4.1f}s) -> " + " | ".join(acc_strs), flush=True)

    total_time_min = (time.time() - t_start_group) / 60.0

    print("\n" + "=" * 88)
    print(f"🎉 GRUPO {args.mode.upper()} CONCLUÍDO EM {total_time_min:.1f} MINUTOS!")
    print("=" * 88)

    # Salvar CSV para CADA modelo do grupo
    for lm in loaded_models:
        df_res = pd.DataFrame(lm["results"])
        df_res.to_csv(lm["out_csv"], index=False)
        overall_acc = (lm["total_detected_all"] / lm["total_imgs_all"]) * 100 if lm["total_imgs_all"] > 0 else 0
        print(f"💾 Salvo: {lm['out_csv'].name} | Acerto Geral: {overall_acc:.2f}% | Total Fakes: {lm['total_imgs_all']:,}")

    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
