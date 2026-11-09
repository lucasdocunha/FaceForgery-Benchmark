"""Benchmark e Estudo Experimental de DTCWT (Dual-Tree Wavelets) e SRM (Spatial Rich Models).

Comparações realizadas:
1. Decomposição Direcional DTCWT (6 sub-bandas em +-15°, +-45°, +-75°) vs DWT clássica (Haar).
2. Resíduos de Alta Frequência SRM (1ª ordem, 2ª ordem Laplaciano e 3ª ordem quadrado).
3. Estatísticas Forenses de Separação (Energia HF, Curtose de Ruído, Anisotropia Direcional).
4. Adaptação dos Modelos do Repositório (ResNet-18, MobileNetV3, Xception) para entradas multicanais:
   - Baseline RGB (3 canais)
   - SRM Enriquecido (6 canais: RGB + 3 SRM)
   - DTCWT Enriquecido (9 canais: RGB + 6 DTCWT)
"""

from __future__ import annotations
import os
import io
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import pywt

from src.data.paths import models_root
from src.pipelines.checkpoints import discover_trained_runs, load_model_from_run
from src.models._channel_adapt import adapt_conv2d_channels
from src.forensics.srm import SRMConv2d
from src.forensics.dtcwt_module import DTCWTExtractor


def compute_forensic_metrics(img_tensor: torch.Tensor, srm_tensor: torch.Tensor, dtcwt_tensor: torch.Tensor) -> dict:
    """Calcula estatísticas de sinal forense para quantificar artefatos de manipulação."""
    # img_tensor: [3, H, W]
    # srm_tensor: [3, H, W]
    # dtcwt_tensor: [6, H, W]
    
    # 1. Energia dos resíduos SRM (sum of squared residuals)
    srm_np = srm_tensor.cpu().numpy()
    srm_energy = float(np.mean(srm_np ** 2))
    
    # 2. Curtose dos resíduos (indica caudas pesadas / artefatos esparsos de costura)
    srm_flat = srm_np.flatten()
    srm_kurtosis = float(np.mean((srm_flat - np.mean(srm_flat))**4) / (np.var(srm_flat)**2 + 1e-12))
    
    # 3. Energia Direcional DTCWT por orientação
    dtcwt_np = dtcwt_tensor.cpu().numpy()  # [6, H, W]
    band_energies = [float(np.mean(dtcwt_np[i] ** 2)) for i in range(6)]
    dtcwt_total_energy = float(np.sum(band_energies))
    
    # 4. Anisotropia direcional: desvio padrão de energia entre as 6 orientações
    # Deepfakes geram descontinuidades direcionais assimétricas em costuras elípticas
    dtcwt_anisotropy = float(np.std(band_energies) / (np.mean(band_energies) + 1e-12))

    return {
        "srm_energy": srm_energy,
        "srm_kurtosis": srm_kurtosis,
        "dtcwt_total_energy": dtcwt_total_energy,
        "dtcwt_anisotropy": dtcwt_anisotropy,
        "energy_15deg": band_energies[0],
        "energy_45deg": band_energies[1],
        "energy_75deg": band_energies[2],
        "energy_m75deg": band_energies[3],
        "energy_m45deg": band_energies[4],
        "energy_m15deg": band_energies[5],
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Benchmark Experimental: DTCWT & SRM Forense na GPU ({device}) ===", flush=True)

    out_fig_dir = Path("figures/wavelets_srm")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_table_dir = Path("tables")
    out_table_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")
    samples = {
        "real_1": {"path": crops_dir / "00208_f00028.jpg", "label": "Face Real 1 (YouTube-real)", "type": "Real"},
        "real_2": {"path": crops_dir / "00011_f00088.jpg", "label": "Face Real 2 (Celeb-real)", "type": "Real"},
        "fake_1": {"path": crops_dir / "id1_id0_0007_f00020.jpg", "label": "DeepFake 1 (Celeb-synthesis)", "type": "Fake"},
        "fake_2": {"path": crops_dir / "id0_id16_0003_f00066.jpg", "label": "DeepFake 2 (Celeb-synthesis)", "type": "Fake"},
    }

    srm_extractor = SRMConv2d(mode="residual_only").to(device)
    dtcwt_extractor = DTCWTExtractor(mode="directional_only").to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    extracted_data = {}
    metrics_rows = []

    print("\n--- 1. Extração de Sinais Forenses (SRM e DTCWT) ---", flush=True)
    for s_key, s_info in samples.items():
        print(f"Processando amostra {s_info['label']} ...", flush=True)
        img_raw = Image.open(s_info["path"]).convert("RGB").resize((224, 224))
        x_tensor = transforms.ToTensor()(img_raw).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # Extração SRM (3 resíduos de ruído)
        with torch.no_grad():
            srm_res = srm_extractor(x_tensor)[0]  # [3, 224, 224]
            dtcwt_bands = dtcwt_extractor(x_tensor)[0]  # [6, 224, 224]

        # DWT clássica (Haar) via PyWavelets para comparação
        gray_np = (0.299 * x_tensor[0, 0] + 0.587 * x_tensor[0, 1] + 0.114 * x_tensor[0, 2]).cpu().numpy()
        coeffs_dwt = pywt.dwt2(gray_np, 'haar')
        ll, (lh, hl, hh) = coeffs_dwt

        extracted_data[s_key] = {
            "img_raw": img_raw,
            "x_tensor": x_tensor[0],
            "srm": srm_res,
            "dtcwt": dtcwt_bands,
            "dwt": {"ll": ll, "lh": lh, "hl": hl, "hh": hh}
        }

        # Estatísticas
        metrics = compute_forensic_metrics(x_tensor[0], srm_res, dtcwt_bands)
        metrics["sample_key"] = s_key
        metrics["sample_label"] = s_info["label"]
        metrics["sample_type"] = s_info["type"]
        metrics_rows.append(metrics)

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = out_table_dir / "srm_dtcwt_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[✓] Métricas forenses salvas em: {metrics_csv}")
    print(df_metrics[["sample_key", "sample_type", "srm_energy", "srm_kurtosis", "dtcwt_total_energy", "dtcwt_anisotropy"]].to_string())

    # --- 2. Renderização do Painel 1: Decomposição Direcional DTCWT (6 Ângulos) ---
    print("\n--- 2. Renderizando Painel Direcional DTCWT (6 Ângulos Canônicos) ---", flush=True)
    fig, axes = plt.subplots(4, 7, figsize=(24, 14))
    col_titles = [
        "Face Original",
        "Sub-banda +15°\n(Quase Horizontal)",
        "Sub-banda +45°\n(Diagonal Principal)",
        "Sub-banda +75°\n(Quase Vertical)",
        "Sub-banda -75°\n(Vertical Invertida)",
        "Sub-banda -45°\n(Diagonal Secundária)",
        "Sub-banda -15°\n(Horizontal Invertida)"
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12, fontweight="bold", pad=10)

    rows = [("real_1", "Real 1", 0), ("real_2", "Real 2", 1), ("fake_1", "DeepFake 1", 2), ("fake_2", "DeepFake 2", 3)]
    for s_key, s_name, r_idx in rows:
        axes[r_idx, 0].imshow(extracted_data[s_key]["img_raw"])
        axes[r_idx, 0].set_ylabel(samples[s_key]["label"], fontsize=11, fontweight="bold")
        axes[r_idx, 0].set_xticks([])
        axes[r_idx, 0].set_yticks([])

        bands = extracted_data[s_key]["dtcwt"].cpu().numpy()
        for b_idx in range(6):
            ax = axes[r_idx, b_idx + 1]
            im = ax.imshow(bands[b_idx], cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Decomposição Espectral DTCWT 2D: Resolução Direcional em 6 Sub-bandas Angulares (±15°, ±45°, ±75°)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    p1_path = out_fig_dir / "dtcwt_directional_decomposition.png"
    plt.savefig(p1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Painel DTCWT salvo em: {p1_path}")

    # --- 3. Renderização do Painel 2: Resíduos de Ruído SRM (3 Filtros de Esteganálise) ---
    print("\n--- 3. Renderizando Painel SRM (Spatial Rich Models) ---", flush=True)
    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    srm_titles = [
        "Face Original",
        "SRM 1: Gradiente 1ª Ordem\n(Borda Horizontal)",
        "SRM 2: Laplaciano 2ª Ordem\n(Descontinuidade Local)",
        "SRM 3: Quadrado 3ª Ordem\n(Resíduo Rico de Esteganálise)"
    ]
    for c, t in enumerate(srm_titles):
        axes[0, c].set_title(t, fontsize=12, fontweight="bold", pad=10)

    for s_key, s_name, r_idx in rows:
        axes[r_idx, 0].imshow(extracted_data[s_key]["img_raw"])
        axes[r_idx, 0].set_ylabel(samples[s_key]["label"], fontsize=11, fontweight="bold")
        axes[r_idx, 0].set_xticks([])
        axes[r_idx, 0].set_yticks([])

        srm_res = extracted_data[s_key]["srm"].cpu().numpy()
        for f_idx in range(3):
            ax = axes[r_idx, f_idx + 1]
            ax.imshow(srm_res[f_idx], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Extração de Resíduos Forenses SRM (Spatial Rich Models): Supressão de Conteúdo e Isolamento de Ruído",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    p2_path = out_fig_dir / "srm_noise_residuals.png"
    plt.savefig(p2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Painel SRM salvo em: {p2_path}")

    # --- 4. Renderização do Painel 3: DWT Clássica vs DTCWT Direcional ---
    print("\n--- 4. Renderizando Comparativo DWT Clássica vs DTCWT ---", flush=True)
    fig, axes = plt.subplots(2, 6, figsize=(22, 8))
    # Amostra Real 1 vs Amostra Fake 1
    cmp_samples = [("real_1", "Face Real (YouTube-real)", 0), ("fake_1", "DeepFake (Celeb-synthesis)", 1)]
    for s_key, s_title, r_idx in cmp_samples:
        d = extracted_data[s_key]
        # Col 0: Original
        axes[r_idx, 0].imshow(d["img_raw"])
        axes[r_idx, 0].set_title("Original" if r_idx == 0 else "", fontweight="bold")
        axes[r_idx, 0].set_ylabel(s_title, fontsize=11, fontweight="bold")
        
        # Col 1: DWT Diagonal (HH) - mistura +45 e -45
        axes[r_idx, 1].imshow(np.abs(d["dwt"]["hh"]), cmap="magma")
        axes[r_idx, 1].set_title("DWT Padrão (Haar Diagonal)\n[Mistura +45° e -45°]" if r_idx == 0 else "", fontweight="bold")
        
        # Col 2: DTCWT +45° (Diagonal pura 1)
        axes[r_idx, 2].imshow(d["dtcwt"][1].cpu().numpy(), cmap="magma")
        axes[r_idx, 2].set_title("DTCWT (+45° Direcional)\n[Diagonal Pura 1]" if r_idx == 0 else "", fontweight="bold")
        
        # Col 3: DTCWT -45° (Diagonal pura 2)
        axes[r_idx, 3].imshow(d["dtcwt"][4].cpu().numpy(), cmap="magma")
        axes[r_idx, 3].set_title("DTCWT (-45° Direcional)\n[Diagonal Pura 2]" if r_idx == 0 else "", fontweight="bold")
        
        # Col 4: SRM Laplaciano
        axes[r_idx, 4].imshow(d["srm"][1].cpu().numpy(), cmap="gray")
        axes[r_idx, 4].set_title("SRM Laplaciano (2ª Ordem)" if r_idx == 0 else "", fontweight="bold")
        
        # Col 5: SRM Quadrado (3ª Ordem)
        axes[r_idx, 5].imshow(d["srm"][2].cpu().numpy(), cmap="gray")
        axes[r_idx, 5].set_title("SRM Quadrado (3ª Ordem)" if r_idx == 0 else "", fontweight="bold")

        for ax in axes[r_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Superioridade Direcional da DTCWT sobre a DWT Clássica e Resíduos SRM de Esteganálise",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    p3_path = out_fig_dir / "wavelets_dwt_vs_dtcwt_comparison.png"
    plt.savefig(p3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Comparativo DWT vs DTCWT salvo em: {p3_path}")

    # --- 5. Teste de Adaptação dos Modelos em 5 Sementes Canônicas (42, 123, 2024, 7, 2025) ---
    print("\n--- 5. Teste de Adaptação dos Modelos em 5 Sementes Canônicas (42, 123, 2024, 7, 2025) ---", flush=True)
    runs = discover_trained_runs(models_root())
    
    architectures = ["resnet", "mobilenet", "xception"]
    target_seeds = [42, 123, 2024, 7, 2025]
    per_seed_results = []
    summary_results = []

    for arch in architectures:
        print(f"\nAvaliando arquitetura {arch.upper()} nas 5 sementes canônicas...", flush=True)
        arch_fps_rgb = []
        arch_fps_srm = []
        arch_fps_dtcwt = []

        for seed in target_seeds:
            cands = [r for r in runs if r.model_family == arch and r.seed == seed]
            if not cands:
                raise RuntimeError(f"Checkpoint para {arch} seed {seed} não encontrado!")
            r = cands[0]

            # 1. Modelo Baseline RGB (3 canais)
            m_rgb = load_model_from_run(r, device).eval()
            
            # 2. Modelo Adaptado para SRM (6 canais: RGB + 3 SRM)
            m_srm = load_model_from_run(r, device).eval()
            if arch == "resnet":
                m_srm.conv1 = adapt_conv2d_channels(m_srm.conv1, 6)
            elif arch == "mobilenet":
                m_srm.features[0][0] = adapt_conv2d_channels(m_srm.features[0][0], 6)
            elif arch == "xception":
                m_srm.backbone.conv1 = adapt_conv2d_channels(m_srm.backbone.conv1, 6)
                
            # 3. Modelo Adaptado para DTCWT (9 canais: RGB + 6 DTCWT)
            m_dtcwt = load_model_from_run(r, device).eval()
            if arch == "resnet":
                m_dtcwt.conv1 = adapt_conv2d_channels(m_dtcwt.conv1, 9)
            elif arch == "mobilenet":
                m_dtcwt.features[0][0] = adapt_conv2d_channels(m_dtcwt.features[0][0], 9)
            elif arch == "xception":
                m_dtcwt.backbone.conv1 = adapt_conv2d_channels(m_dtcwt.backbone.conv1, 9)

            dummy_rgb = torch.randn(16, 3, 224, 224, device=device)
            dummy_srm = torch.randn(16, 6, 224, 224, device=device)
            dummy_dtcwt = torch.randn(16, 9, 224, 224, device=device)

            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = m_rgb(dummy_rgb)
                    _ = m_srm(dummy_srm)
                    _ = m_dtcwt(dummy_dtcwt)
                torch.cuda.synchronize()

                # RGB
                t0 = time.time()
                for _ in range(15):
                    _ = m_rgb(dummy_rgb)
                torch.cuda.synchronize()
                fps_rgb = (16 * 15) / (time.time() - t0)

                # SRM
                t0 = time.time()
                for _ in range(15):
                    _ = m_srm(dummy_srm)
                torch.cuda.synchronize()
                fps_srm = (16 * 15) / (time.time() - t0)

                # DTCWT
                t0 = time.time()
                for _ in range(15):
                    _ = m_dtcwt(dummy_dtcwt)
                torch.cuda.synchronize()
                fps_dtcwt = (16 * 15) / (time.time() - t0)

            arch_fps_rgb.append(fps_rgb)
            arch_fps_srm.append(fps_srm)
            arch_fps_dtcwt.append(fps_dtcwt)

            per_seed_results.append({
                "architecture": arch.upper(),
                "seed": seed,
                "rgb_fps": round(fps_rgb, 1),
                "srm_6c_fps": round(fps_srm, 1),
                "dtcwt_9c_fps": round(fps_dtcwt, 1),
                "overhead_srm_pct": round(((fps_rgb - fps_srm) / fps_rgb) * 100, 2),
                "overhead_dtcwt_pct": round(((fps_rgb - fps_dtcwt) / fps_rgb) * 100, 2),
            })
            print(f"  [Seed {seed:4d}] RGB: {fps_rgb:.1f} FPS | SRM: {fps_srm:.1f} FPS | DTCWT: {fps_dtcwt:.1f} FPS")

        mean_rgb, std_rgb = np.mean(arch_fps_rgb), np.std(arch_fps_rgb)
        mean_srm, std_srm = np.mean(arch_fps_srm), np.std(arch_fps_srm)
        mean_dtcwt, std_dtcwt = np.mean(arch_fps_dtcwt), np.std(arch_fps_dtcwt)
        ovh_srm = ((mean_rgb - mean_srm) / mean_rgb) * 100
        ovh_dtcwt = ((mean_rgb - mean_dtcwt) / mean_rgb) * 100

        summary_results.append({
            "architecture": arch.upper(),
            "seeds": "5 seeds (42, 123, 2024, 7, 2025)",
            "rgb_fps_mean": round(mean_rgb, 1),
            "rgb_fps_std": round(std_rgb, 1),
            "srm_6c_fps_mean": round(mean_srm, 1),
            "srm_6c_fps_std": round(std_srm, 1),
            "dtcwt_9c_fps_mean": round(mean_dtcwt, 1),
            "dtcwt_9c_fps_std": round(std_dtcwt, 1),
            "overhead_srm_pct": round(ovh_srm, 2),
            "overhead_dtcwt_pct": round(ovh_dtcwt, 2),
        })

    df_per_seed = pd.DataFrame(per_seed_results)
    df_per_seed.to_csv(out_table_dir / "srm_dtcwt_5seeds_benchmark.csv", index=False)

    df_summary = pd.DataFrame(summary_results)
    models_csv = out_table_dir / "srm_dtcwt_model_benchmarks.csv"
    df_summary.to_csv(models_csv, index=False)
    print(f"\n[✓] Benchmarks consolidados nas 5 sementes salvos em: {models_csv}")
    print(df_summary.to_string())

    print("\n=======================================================")
    print(" [✓] BENCHMARK EXPERIMENTAL DE DTCWT & SRM CONCLUÍDO!")
    print(f"  - Gráficos gerados em: {out_fig_dir}/")
    print(f"  - Tabelas geradas em: {out_table_dir}/")
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
