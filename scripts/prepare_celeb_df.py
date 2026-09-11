"""Extração de frames e recorte facial para o benchmark Celeb-DF v2.

Lê List_of_testing_videos.txt, seleciona N frames uniformemente espaçados por vídeo,
detecta a face principal, aplica padding de contexto e salva crops 512x512 em disco,
gerando o CSV de manifesto de teste compatível com ImageDataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_testing_list(list_path: Path, dataset_root: Path) -> list[tuple[int, Path, str]]:
    """Lê List_of_testing_videos.txt.
    
    Formato: <label> <relative_path>
    Onde label 1 = Real, 0 = Fake/Synthesis.
    """
    entries = []
    if not list_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {list_path}")

    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            label = int(parts[0])
            rel_path = parts[1]
            full_path = dataset_root / rel_path
            video_id = Path(rel_path).stem
            entries.append((label, full_path, video_id))
    return entries


def crop_face_or_center(img: cv2.typing.MatLike, cascade: cv2.CascadeClassifier, target_size: int = 512) -> Image.Image:
    """Detecta face principal com padding; se não encontrar, usa crop central."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]

        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        cw = x2 - x1
        ch = y2 - y1
        diff = abs(cw - ch)
        if cw < ch:
            pad = diff // 2
            x1 = max(0, x1 - pad)
            x2 = min(w, x2 + (diff - pad))
        elif ch < cw:
            pad = diff // 2
            y1 = max(0, y1 - pad)
            y2 = min(h, y2 + (diff - pad))

        crop = img[y1:y2, x1:x2]
    else:
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = img[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    return pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)


def process_single_video(
    label: int,
    video_path: str,
    video_id: str,
    output_dir: str,
    frames_per_video: int,
    target_size: int,
) -> list[dict]:
    video_p = Path(video_path)
    out_dir = Path(output_dir)
    if not video_p.exists():
        return []

    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    if total_frames <= frames_per_video:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / (frames_per_video + 1)
        frame_indices = [int(step * (i + 1)) for i in range(frames_per_video)]

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    records = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        face_pil = crop_face_or_center(frame, cascade, target_size=target_size)
        save_name = f"{video_id}_f{idx:05d}.jpg"
        save_path = out_dir / save_name
        face_pil.save(save_path, format="JPEG", quality=95)

        records.append({
            "img_name": save_name,
            "target": label,
            "video_id": video_id,
            "frame_idx": idx,
        })

    cap.release()
    return records


def main():
    parser = argparse.ArgumentParser(description="Extração e recorte de faces do Celeb-DF v2")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/media/ssd2/lucas.ocunha/datasets"),
        help="Raiz onde estão os vídeos e List_of_testing_videos.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops"),
        help="Diretório onde as imagens recortadas serão salvas",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/celeb_df/test.csv"),
        help="Caminho para o CSV de manifesto de saída",
    )
    parser.add_argument("--frames-per-video", type=int, default=15, help="Número de frames por vídeo")
    parser.add_argument("--target-size", type=int, default=512, help="Resolução das faces recortadas")
    parser.add_argument("--num-workers", type=int, default=8, help="Processos paralelos para extração")
    parser.add_argument("--limit-videos", type=int, default=None, help="Limita o número de vídeos (smoke test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    list_path = args.dataset_root / "List_of_testing_videos.txt"
    entries = parse_testing_list(list_path, args.dataset_root)
    if args.limit_videos:
        entries = entries[: args.limit_videos]

    print(f"Total de vídeos para processar: {len(entries)}")
    print(f"Extraindo {args.frames_per_video} frames por vídeo para: {args.output_dir}")

    all_records = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_video,
                label,
                str(vpath),
                vid,
                str(args.output_dir),
                args.frames_per_video,
                args.target_size,
            ): vid
            for label, vpath, vid in entries
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processando vídeos"):
            res = fut.result()
            all_records.extend(res)

    df = pd.DataFrame(all_records)
    df.to_csv(args.manifest_out, index=False)
    print(f"\nManifesto salvo em: {args.manifest_out}")
    print(f"Total de imagens salvas: {len(df)}")
    if not df.empty:
        print("Distribuição de classes:")
        print(df["target"].value_counts().rename({1: "Real (1)", 0: "Fake (0)"}))


if __name__ == "__main__":
    main()
