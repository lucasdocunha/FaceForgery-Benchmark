#!/usr/bin/env python
"""
Download FaceForensics++ (FF++) public data release.
Configurado para o benchmark canônico:
  - Métodos: original (YouTube), Deepfakes, Face2Face, FaceSwap, NeuralTextures
  - Compressão: c23 (HQ / H.264 constante - padrão da literatura)
  - Tipo: videos
  - Servidor: EU2 (https://kaldir.vc.in.tum.de/faceforensics/)
  - Download multithreaded paralelo com suporte a retomada (resume).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join, isfile, getsize
from pathlib import Path
from tqdm import tqdm

# URLs e arquivos de metadados
FILELIST_URL = 'misc/filelist.json'
DEEPFAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'

DATASETS = {
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

CANONICAL_DATASETS = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
ALL_DATASETS = list(DATASETS.keys())
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU2', 'EU', 'CA']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics++ public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--output_path', type=str, default='/media/ssd2/datasets/FaceForensics',
        help='Output directory.'
    )
    parser.add_argument(
        '-d', '--dataset', type=str, default='canonical',
        help="Datasets a baixar: 'canonical' (original + 4 métodos), 'all' ou nome específico.",
        choices=['canonical', 'all'] + list(DATASETS.keys())
    )
    parser.add_argument(
        '-c', '--compression', type=str, default='c23',
        help='Grau de compressão (c23 = padrão da literatura, c40 = baixa qualidade, raw = lossless).',
        choices=COMPRESSION
    )
    parser.add_argument(
        '-t', '--type', type=str, default='videos',
        help='Tipo de arquivo (videos, masks, models).',
        choices=TYPE
    )
    parser.add_argument(
        '-n', '--num_videos', type=int, default=None,
        help='Número de vídeos para baixar (None = baixar todos os 1000 por método).'
    )
    parser.add_argument(
        '--server', type=str, default='EU2',
        help='Servidor TUM (EU2 = kaldir HTTPS ativo).',
        choices=SERVERS
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Número de threads de download concorrentes.'
    )
    parser.add_argument(
        '--retries', type=int, default=3,
        help='Número de tentativas por arquivo em caso de erro de rede.'
    )

    args = parser.parse_args()

    # URLs dos servidores
    if args.server == 'EU2':
        server_url = 'https://kaldir.vc.in.tum.de/faceforensics/'
    elif args.server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif args.server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise ValueError(f'Servidor desconhecido: {args.server}')

    args.base_url = server_url + 'v3/'
    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    return args


def download_single_file(url: str, out_file: str, max_retries: int = 3) -> tuple[str, bool, str]:
    """Baixa um único arquivo de forma atômica com retentativas."""
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    # Verifica se arquivo já existe e é válido
    if isfile(out_file) and getsize(out_file) > 1024:
        return out_file, True, "exists"

    for attempt in range(1, max_retries + 1):
        tmp_path = None
        try:
            fh, tmp_path = tempfile.mkstemp(dir=out_dir, prefix='.ff_tmp_')
            os.close(fh)

            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'FaceForensics-Downloader/2.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, 'wb') as f_out:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)

            # Verifica integridade básica
            if getsize(tmp_path) > 0:
                os.rename(tmp_path, out_file)
                try:
                    os.chmod(out_file, 0o664)
                except OSError:
                    pass
                return out_file, True, "downloaded"
            else:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt == max_retries:
                return out_file, False, str(e)
            time.sleep(1.0 * attempt)

    return out_file, False, "unknown_error"


def main():
    args = parse_args()

    print("=" * 70)
    print(" FaceForensics++ Dataset Downloader (Canônico c23)")
    print("=" * 70)
    print(f" Servidor Base:  {args.base_url}")
    print(f" Destino:        {args.output_path}")
    print(f" Compressão:     {args.compression}")
    print(f" Tipo:           {args.type}")
    print(f" Threads:        {args.workers}")
    print("=" * 70)

    # Define lista de datasets
    if args.dataset == 'canonical':
        c_datasets = CANONICAL_DATASETS
    elif args.dataset == 'all':
        c_datasets = ALL_DATASETS
    else:
        c_datasets = [args.dataset]

    os.makedirs(args.output_path, exist_ok=True)

    # Obtém filelist central da TUM
    filelist_url = args.base_url + FILELIST_URL
    print(f"\nObtendo lista oficial de arquivos da TUM ({filelist_url})...")
    req = urllib.request.Request(filelist_url, headers={'User-Agent': 'FaceForensics-Downloader/2.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        file_pairs = json.loads(resp.read().decode('utf-8'))

    print(f"Total de pares de vídeo indexados: {len(file_pairs)}")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for ds_name in c_datasets:
        ds_rel = DATASETS[ds_name]
        print(f"\n>>> Processando dataset: '{ds_name}' ({ds_rel})...")

        # Constrói a lista de arquivos para este dataset
        filelist = []
        if ds_name == 'original':
            for pair in file_pairs:
                filelist.extend(pair)
            # Remove duplicatas mantendo ordem
            seen = set()
            filelist = [x for x in filelist if not (x in seen or seen.add(x))]
        else:
            for pair in file_pairs:
                filelist.append(f"{pair[0]}_{pair[1]}")
                filelist.append(f"{pair[1]}_{pair[0]}")

        if args.num_videos is not None and args.num_videos > 0:
            filelist = filelist[:args.num_videos]

        # URLs e pastas de saída
        base_video_url = f"{args.base_url}{ds_rel}/{args.compression}/{args.type}/"
        out_dir = join(args.output_path, ds_rel, args.compression, args.type)
        os.makedirs(out_dir, exist_ok=True)

        tasks = []
        for filename in filelist:
            v_name = f"{filename}.mp4"
            v_url = base_video_url + v_name
            v_out = join(out_dir, v_name)
            tasks.append((v_url, v_out))

        print(f"  Arquivos a verificar/baixar: {len(tasks)}")
        print(f"  Diretório local: {out_dir}")

        ds_downloaded = 0
        ds_skipped = 0
        ds_failed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(download_single_file, url, out, args.retries): (url, out)
                for url, out in tasks
            }

            pbar = tqdm(total=len(tasks), desc=f"  {ds_name}", unit="vid", dynamic_ncols=True)
            for future in as_completed(future_to_task):
                out_file, success, status = future.result()
                if success:
                    if status == "exists":
                        ds_skipped += 1
                    else:
                        ds_downloaded += 1
                else:
                    ds_failed += 1
                    tqdm.write(f"  [ERRO] Falha ao baixar {out_file}: {status}")
                pbar.update(1)
            pbar.close()

        print(f"  Resultado '{ds_name}': {ds_downloaded} baixados, {ds_skipped} já existentes, {ds_failed} falhas.")
        total_downloaded += ds_downloaded
        total_skipped += ds_skipped
        total_failed += ds_failed

    print("\n" + "=" * 70)
    print(" DOWNLOAD DO FACEFORENSICS++ FINALIZADO")
    print(f" Baixados com sucesso: {total_downloaded}")
    print(f" Já existentes (skip):  {total_skipped}")
    print(f" Falhas:                {total_failed}")
    print(f" Pasta compartilhada:   {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
