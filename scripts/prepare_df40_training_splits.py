"""Prepara os splits balanceados de treino e validação do DF40 sem vazamento do test.csv."""

import random
from pathlib import Path
import pandas as pd

def main():
    random.seed(42)
    root = Path("/media/ssd2/lucas.ocunha/datasets/df40_extracted")
    test_csv = Path("data/df40/test.csv")
    assert test_csv.exists(), "data/df40/test.csv não encontrado!"

    test_df = pd.read_csv(test_csv)
    banned = set(Path(p).resolve() for p in test_df.iloc[:, 0])
    print(f"Total amostras no test.csv: {len(banned)}")

    # 1. Coleta das imagens REAIS (fora do teste)
    real_dirs = [
        root / "whichfaceisreal/real",
        root / "CollabDiff/real",
        root / "MidJourney/real",
        root / "starganv2/real",
        root / "styleclip/real",
        root / "heygen_new/real",
        root / "stargan/real",
        root / "deepfacelab/real",
    ]

    clean_reals = []
    for d in real_dirs:
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in (".jpg", ".png", ".jpeg") and not p.name.startswith("._") and "__MACOSX" not in str(p):
                    if p.stat().st_size > 1024:
                        rp = p.resolve()
                        if rp not in banned:
                            clean_reals.append(str(rp))

    clean_reals = sorted(list(set(clean_reals)))
    random.shuffle(clean_reals)
    print(f"Total de imagens REAIS disponíveis: {len(clean_reals)}")

    # 2. Coleta das imagens FAKE por categoria
    fake_categories = [
        'DiT', 'SiT', 'ddim', 'sd2.1', 'pixart', 'RDDM', 'CollabDiff/fake',
        'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'VQGAN', 'stargan/fake', 'starganv2/fake',
        'deepfacelab/fake', 'faceswap', 'mobileswap', 'blendface', 'uniface',
        'MidJourney/fake', 'whichfaceisreal/fake', 'styleclip/fake', 'e4e',
        'heygen_new/fake', 'cdf/frames'
    ]

    fakes_by_cat = {}
    for cat in fake_categories:
        folder = root / cat
        cat_fakes = []
        if folder.exists():
            for p in folder.rglob("*"):
                if p.suffix.lower() in (".jpg", ".png", ".jpeg") and not p.name.startswith("._") and "__MACOSX" not in str(p):
                    if p.stat().st_size > 1024:
                        rp = p.resolve()
                        if rp not in banned:
                            cat_fakes.append(str(rp))
        cat_fakes = sorted(list(set(cat_fakes)))
        random.shuffle(cat_fakes)
        fakes_by_cat[cat] = cat_fakes

    # Definir alvos:
    # Treino: 8.000 reals + 8.000 fakes = 16.000
    # Val: 2.000 reals + 2.000 fakes = 4.000
    n_train_reals = 8000
    n_val_reals = 2000
    train_reals = clean_reals[:n_train_reals]
    val_reals = clean_reals[n_train_reals:n_train_reals + n_val_reals]

    # Amostragem estratificada dos fakes
    train_fakes = []
    val_fakes = []

    # Meta de amostras por categoria
    target_train_per_cat = n_train_reals // len(fake_categories) # ~333
    target_val_per_cat = n_val_reals // len(fake_categories)     # ~83

    # Para categorias com poucas imagens (ex: MidJourney/fake tem 387), pegamos proporcional
    for cat, flist in fakes_by_cat.items():
        n_avail = len(flist)
        n_val = min(target_val_per_cat, int(n_avail * 0.2))
        n_train = min(target_train_per_cat, n_avail - n_val)
        
        val_fakes.extend(flist[:n_val])
        train_fakes.extend(flist[n_val:n_val + n_train])

    # Se faltar um pouco para atingir 8000 e 2000, completa das categorias com abundância (DiT, sd2.1, etc.)
    large_cats = ['DiT', 'SiT', 'sd2.1', 'pixart', 'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'faceswap', 'mobileswap', 'e4e', 'cdf/frames']
    pool_train = []
    pool_val = []
    for cat in large_cats:
        flist = fakes_by_cat[cat]
        # Pega amostras após as já alocadas
        allocated = target_train_per_cat + target_val_per_cat
        pool_train.extend(flist[allocated:])
    
    random.shuffle(pool_train)
    deficit_train = n_train_reals - len(train_fakes)
    if deficit_train > 0:
        train_fakes.extend(pool_train[:deficit_train])

    deficit_val = n_val_reals - len(val_fakes)
    if deficit_val > 0:
        val_fakes.extend(pool_train[deficit_train:deficit_train + deficit_val])

    print(f"Treino montado: {len(train_reals)} Reais, {len(train_fakes)} Fakes (Total: {len(train_reals) + len(train_fakes)})")
    print(f"Validação montada: {len(val_reals)} Reais, {len(val_fakes)} Fakes (Total: {len(val_reals) + len(val_fakes)})")

    # Montar DataFrames
    train_df = pd.DataFrame([
        {"img_name": p, "target": 0} for p in train_reals
    ] + [
        {"img_name": p, "target": 1} for p in train_fakes
    ])
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_df = pd.DataFrame([
        {"img_name": p, "target": 0} for p in val_reals
    ] + [
        {"img_name": p, "target": 1} for p in val_fakes
    ])
    val_df = val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Verificações de sanidade rigorosas
    train_set = set(Path(p).resolve() for p in train_df["img_name"])
    val_set = set(Path(p).resolve() for p in val_df["img_name"])

    assert len(train_set.intersection(banned)) == 0, "ERRO CRÍTICO: Vazamento de teste no treino!"
    assert len(val_set.intersection(banned)) == 0, "ERRO CRÍTICO: Vazamento de teste na validação!"
    assert len(train_set.intersection(val_set)) == 0, "ERRO CRÍTICO: Vazamento de treino na validação!"

    out_train = Path("data/df40/train.csv")
    out_val = Path("data/df40/val.csv")
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    print(f"Salvos com sucesso:\n - {out_train} ({len(train_df)} linhas)\n - {out_val} ({len(val_df)} linhas)")

if __name__ == "__main__":
    main()
