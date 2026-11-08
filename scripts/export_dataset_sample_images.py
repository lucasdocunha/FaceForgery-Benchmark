"""Script para exportar amostras individuais das bases MFFI, Celeb-DF v2 e DF-40
no padrão solicitado para inserção em documentos LaTeX/Overleaf:
    imagem_{base}_{classe}_{idx}.png
"""

import os
import shutil
import pandas as pd
from PIL import Image

def export_samples():
    output_dir = "figures/amostras"
    os.makedirs(output_dir, exist_ok=True)

    # 1. MFFI
    df_mffi = pd.read_csv("data/manifests/test.csv")
    root_mffi = "data/datasets/phase1/testset"
    mffi_reals = [
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 0].iloc[0]["img_name"]),
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 0].iloc[1]["img_name"]),
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 0].iloc[2]["img_name"]),
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 0].iloc[3]["img_name"]),
    ]
    mffi_fakes = [
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 1].iloc[0]["img_name"]),
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 1].iloc[1]["img_name"]),
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 1].iloc[2]["img_name"]),
        os.path.join(root_mffi, df_mffi[df_mffi["label"] == 1].iloc[3]["img_name"]),
    ]

    # 2. Celeb-DF
    df_celeb = pd.read_csv("data/celeb_df/test.csv")
    root_celeb = "data/datasets/celeb_df_crops"
    celeb_reals = [
        os.path.join(root_celeb, "id1_0007_f00020.jpg"),     # Celeb-real
        os.path.join(root_celeb, "00208_f00028.jpg"),        # YouTube-real
        os.path.join(root_celeb, "00063_f00029.jpg"),        # YouTube-real
        os.path.join(root_celeb, "00236_f00029.jpg"),        # YouTube-real
    ]
    celeb_fakes = [
        os.path.join(root_celeb, "id1_id0_0007_f00020.jpg"), # id1 -> id0
        os.path.join(root_celeb, "id3_id1_0001_f00023.jpg"), # id3 -> id1
        os.path.join(root_celeb, "id2_id0_0008_f00032.jpg"), # id2 -> id0
        os.path.join(root_celeb, "id2_id3_0008_f00032.jpg"), # id2 -> id3
    ]

    # 3. DF-40
    df_df40 = pd.read_csv("data/df40/test.csv")
    df40_reals = [
        df_df40[df_df40["method"] == "Real_whichfaceisreal"].iloc[0]["img_name"],
        df_df40[df_df40["method"] == "Real_heygen"].iloc[0]["img_name"],
        df_df40[df_df40["method"] == "Real_starganv2"].iloc[0]["img_name"],
        df_df40[df_df40["method"] == "Real_MidJourney"].iloc[0]["img_name"],
    ]
    df40_fakes = [
        df_df40[df_df40["method"] == "faceswap"].iloc[0]["img_name"],
        df_df40[df_df40["method"] == "StyleGAN3"].iloc[0]["img_name"],
        df_df40[df_df40["method"] == "uniface"].iloc[0]["img_name"],
        df_df40[df_df40["method"] == "pixart"].iloc[0]["img_name"],
    ]

    collection = {
        "mffi": {"real": mffi_reals, "fake": mffi_fakes},
        "celeb": {"real": celeb_reals, "fake": celeb_fakes},
        "df40": {"real": df40_reals, "fake": df40_fakes},
    }

    manifest_rows = []

    for base, classes in collection.items():
        for cls_name, paths in classes.items():
            for idx, p in enumerate(paths, start=1):
                # Carrega e converte para RGB
                im = Image.open(p).convert("RGB")
                
                # Nome padrão com 'fake'
                out_name_fake = f"imagem_{base}_{cls_name}_{idx}.png"
                out_path_fake = os.path.join(output_dir, out_name_fake)
                im.save(out_path_fake, "PNG", optimize=True)

                # Se a classe for fake, salva também como 'falso' para conveniência no Overleaf em PT
                if cls_name == "fake":
                    out_name_falso = f"imagem_{base}_falso_{idx}.png"
                    out_path_falso = os.path.join(output_dir, out_name_falso)
                    im.save(out_path_falso, "PNG", optimize=True)

                # Se a base for celeb, cria alias celebdf também
                if base == "celeb":
                    out_name_celebdf = f"imagem_celebdf_{cls_name}_{idx}.png"
                    out_path_celebdf = os.path.join(output_dir, out_name_celebdf)
                    im.save(out_path_celebdf, "PNG", optimize=True)
                    if cls_name == "fake":
                        out_name_celebdf_falso = f"imagem_celebdf_falso_{idx}.png"
                        im.save(os.path.join(output_dir, out_name_celebdf_falso), "PNG", optimize=True)

                manifest_rows.append({
                    "base": base,
                    "classe": cls_name,
                    "indice": idx,
                    "arquivo": out_name_fake,
                    "caminho_origem": p,
                    "largura": im.size[0],
                    "altura": im.size[1],
                })
                print(f"Salvo: {out_name_fake} (de {os.path.basename(p)}, {im.size[0]}x{im.size[1]})")

    df_out = pd.DataFrame(manifest_rows)
    df_out.to_csv(os.path.join(output_dir, "manifest_amostras.csv"), index=False)
    print("\nManifesto salvo em:", os.path.join(output_dir, "manifest_amostras.csv"))
    print(f"Total de amostras originais processadas: {len(manifest_rows)}")

if __name__ == "__main__":
    export_samples()
