import sys, random
from pathlib import Path
from ensemble_fusion import load_metrics_index, _eval_combo, _model_label

SEED = 42
AUC_THRESHOLD = 0.90
SPLITS = ["val", "test_d", "test"]
MODELS_DIR = Path("models")

df = load_metrics_index(MODELS_DIR)
pool = df[(df["model_family"] != "ensemble") & (df["auc"] >= AUC_THRESHOLD)].reset_index(drop=True)

print(f"Pool com AUC >= {AUC_THRESHOLD}: {len(pool)} modelos")
for _, r in pool.sort_values("auc", ascending=False).iterrows():
    print(f"  {_model_label(r.to_dict()):<40}  AUC={r['auc']:.4f}")

rng = random.Random(SEED)
indices = rng.sample(range(len(pool)), 3)
model_rows = [pool.iloc[i].to_dict() for i in indices]

print("\n=== Modelos selecionados ===")
for r in model_rows:
    print(f"  {_model_label(r)}  AUC_val={r['auc']:.4f}")

print("\nFusionando...")
metrics = _eval_combo(model_rows, SPLITS, MODELS_DIR)

if metrics is None:
    print("ERRO: fusão falhou")
    sys.exit(1)

print("\n=== Resultados ===")
for split in SPLITS:
    if f"{split}_auc" in metrics:
        print(
            f"  [{split:8}]  AUC={metrics[f'{split}_auc']:.4f}  "
            f"ACC={metrics[f'{split}_acc']:.4f}  F1={metrics[f'{split}_f1']:.4f}  "
            f"Recall={metrics[f'{split}_recall']:.4f}  Spec={metrics[f'{split}_specificity']:.4f}"
        )
print(f"\n  threshold (val): {metrics['threshold']:.4f}")
