"""Descompacta todos os arquivos zip do DF40 em /media/ssd2/lucas.ocunha/datasets/df40_extracted."""

from __future__ import annotations

import zipfile
import io
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_zip(zip_path: Path, dest_dir: Path) -> str:
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
        count = len(zf.namelist())
    dt = time.time() - t0
    return f"Extraído {zip_path.name} ({count} itens) em {dt:.1f}s"


def extract_chunk_and_inner(chunk_path: Path, dest_dir: Path) -> list[str]:
    results = []
    t0 = time.time()
    with zipfile.ZipFile(chunk_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".zip"):
                t_sub = time.time()
                sub_bytes = zf.read(name)
                with zipfile.ZipFile(io.BytesIO(sub_bytes), "r") as sub_zf:
                    sub_zf.extractall(dest_dir)
                    sub_count = len(sub_zf.namelist())
                results.append(f"  -> {name} ({sub_count} itens) em {time.time() - t_sub:.1f}s")
            else:
                zf.extract(name, dest_dir)
    dt = time.time() - t0
    results.insert(0, f"Processado chunk {chunk_path.name} em {dt:.1f}s")
    return results


def main():
    src_dir = Path("/media/ssd2/lucas.ocunha/datasets/df40")
    dest_dir = Path("/media/ssd2/lucas.ocunha/datasets/df40_extracted")
    dest_dir.mkdir(parents=True, exist_ok=True)

    direct_zips = [
        "DiT-002.zip", "RDDM-004.zip", "SiT-020.zip",
        "VQGAN-008.zip", "ddim-013.zip", "e4e-006.zip", "heygen-007.zip",
        "mobileswap-017.zip", "pixart-021.zip", "sd2.1-005.zip"
    ]
    # StyleGAN2-001.zip was already extracted during the test

    chunk_zips = sorted(list(src_dir.glob("DF40-*.zip")) + [src_dir / "sd2.1-011.zip"])

    print(f"Iniciando descompactação de DF40 em {dest_dir}...", flush=True)
    t_start = time.time()

    # 1. Direct zips
    print(f"\n--- 1/3: Extraindo {len(direct_zips)} zips diretos ---", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(extract_zip, src_dir / z, dest_dir): z
            for z in direct_zips if (src_dir / z).exists()
        }
        for fut in as_completed(futures):
            z_name = futures[fut]
            try:
                msg = fut.result()
                print(msg, flush=True)
            except Exception as e:
                print(f"ERRO em {z_name}: {e}", flush=True)

    # 2. Chunk zips with inner zips
    print(f"\n--- 2/3: Extraindo {len(chunk_zips)} chunks do Google Drive com inner zips ---", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(extract_chunk_and_inner, cz, dest_dir): cz.name
            for cz in chunk_zips if cz.exists()
        }
        for fut in as_completed(futures):
            cz_name = futures[fut]
            try:
                msgs = fut.result()
                for m in msgs:
                    print(m, flush=True)
            except Exception as e:
                print(f"ERRO em {cz_name}: {e}", flush=True)

    # 3. Check for nested zips like mobileswap frames.zip
    print("\n--- 3/3: Verificando e descompactando zips aninhados residuais ---", flush=True)
    nested_zips = list(dest_dir.rglob("*.zip"))
    for nz in nested_zips:
        print(f"Descompactando aninhado {nz}...", flush=True)
        try:
            with zipfile.ZipFile(nz, "r") as zf:
                zf.extractall(nz.parent)
            nz.unlink()
        except Exception as e:
            print(f"Erro ao descompactar {nz}: {e}", flush=True)

    total_time = time.time() - t_start
    print(f"\nDescompactação concluída com sucesso em {total_time/60:.2f} minutos!", flush=True)


if __name__ == "__main__":
    main()
