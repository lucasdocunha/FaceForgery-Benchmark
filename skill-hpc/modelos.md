# Checagem: modelos

## Carregar

1. Primeiro do que o cluster já tem: `/datasets/ai_models/...` (pesos compartilhados, só leitura — ex.: `/datasets/ai_models/ollama`) e `/datasets/containers_ngc/*.sif`.
2. Só se não existir lá: baixar para o **cache local do job** (`${TMPDIR:-/scratch/$USER}/job_$SLURM_JOB_ID`), nunca para o NFS.
3. Modelo próprio já treinado: de `/projects/models/$USER/...`.

## Salvar

```
/projects/models/$USER/<projeto>/<nome>_<AAAA-MM-DD>_<jobid>/
├── MODELO.md    # modelo base, dataset (caminho em /datasets), commit, job id, hiperparâmetros, métricas
└── <pesos>      # save_pretrained / .safetensors / adapter
```

- Checkpoints intermediários no disco local; copiar para `/projects/models` só no fim (ou a cada N checkpoints, mantendo poucos).

## Bloqueia (❌)

- Store próprio de modelo servido (ex.: `OLLAMA_MODELS`) no NFS — deu `EIO`.
- Download de pesos com cache default (`~/.cache/huggingface` = NFS).
- Modelo final salvo em `/home/$USER`, `/tmp`/`/scratch` (some ao fim do job) ou dentro de `/datasets`.

## Aviso (⚠️)

- Baixa da internet um modelo que já existe em `/datasets/ai_models`.
- Modelo salvo sem `MODELO.md` ou sem versão no nome (sobrescreve o anterior).
- Checkpoint salvo a cada step direto no NFS.
- Uso do modelo fora das instruções do cluster/modelo (ex.: `gpu:4` pra modelo que cabe em uma H100; VRAM do modelo > 80GB sem sharding).
