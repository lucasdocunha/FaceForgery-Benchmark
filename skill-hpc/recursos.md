# Checagem: recursos (#SBATCH)

Padrão do usuário (ponto de partida): `gpu:1`, `--mem=64G`, `--cpus-per-task=8`, `--time=48:00:00`, saída em `logs/%x_%j.{out,err}`, `--mail-user=lucas.ocunha@ppgia.pucpr.br`, `--mail-type=ALL`. Ver [template.sh](template.sh).

## Bloqueia (❌)

- `--mem` > 884G, `--cpus-per-task` > 124, `--gres=gpu:N` com N fora de {1, 2, 4}.
- `--partition=shared` com `--gres=gpu:...`, ou `gpu:` e `mps:` juntos, ou `--time` > 4h na `shared`.
- `--partition` com nome que não seja `gpu` ou `shared` (ex.: `gpu:1`).
- `--output`/`--error` em pasta que não existe (`sbatch` não cria) — exigir `mkdir -p logs` antes, em "Comandos pra rodar".
- Código usa GPU (`cuda`, `.to("cuda")`, `device_map`, Ollama/vLLM) e o `.sh` não pede `--gres`.

## Aviso (⚠️) — proporcional ao que o código usa

- `gpu:2`/`gpu:4` sem paralelismo multi-GPU real no código (DDP/`torchrun`/`accelerate`/`device_map="auto"` em modelo que não cabe em 80GB/tensor parallel no vLLM). O cluster pede: homologar com 1 GPU, escalar com prova.
- `--cpus-per-task` muito acima do que o código paraleliza (sem `num_workers`, pool, threads).
- `--mem` muito acima do que o código carrega; ou abaixo (dataset/modelo carregado inteiro em RAM maior que o pedido).
- VRAM estimada do modelo > 80GB por GPU sem sharding.
- Job sem GPU pedindo `--gres`.
- Mail/e-mail diferente do padrão.
