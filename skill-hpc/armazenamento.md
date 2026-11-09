# Checagem: armazenamento

Mapeie **toda** leitura e escrita do código e do `.sh` (open/write, `to_csv`, `save_pretrained`, `torch.save`, `cache_dir`, variáveis `*_HOME`/`*_CACHE`, `mkdir`) e classifique:

| O quê | Onde deve ir |
|---|---|
| código, envs conda | `$HOME` |
| dados de entrada | `/datasets/...` (só leitura no job) — ver [datasets.md](datasets.md) |
| modelos finais | `/projects/models/$USER/...` — ver [modelos.md](modelos.md) |
| saídas do job (predições, CSVs, relatórios) | `<raiz do projeto>/saidas/` |
| cache de runtime (HF, torch, pip, Ollama, checkpoints intermediários) | `${TMPDIR:-/scratch/$USER}/job_$SLURM_JOB_ID`; apagado no `trap EXIT` |
| logs | stdout/stderr → `logs/%x_%j.out` (ver [logs.md](logs.md)) |

Se o projeto grava num banco/servidor externo, esse é o destino dos dados; a tabela vale para arquivos.

## Bloqueia (❌)

- Cache de runtime/blob de modelo no NFS (`$HOME`, `/projects`, `/users`): `HF_HOME`/`TRANSFORMERS_CACHE`/`OLLAMA_MODELS`/`TORCH_HOME` sem definir (default cai em `~/.cache`, que é NFS) ou apontando para o NFS. Escrita binária pesada no NFS já deu `EIO`.
- Escrita em `/home/$USER` (local a um nó) ou em caminho que não existe no cluster.
- Escrita dentro de `/datasets` durante o job.
- Cache em `/scratch` sem limpeza ao sair (disco local compartilhado, 68% cheio).

## Aviso (⚠️)

- Código que gera muitos arquivos numa pasta só (loop salvando 1 arquivo por item): > 1.000 por pasta — organizar em `lote_NNN/`.
- Saída fora de `saidas/` do projeto.
- Download grande (pesos de dezenas de GB) para `/scratch` sem checar espaço livre (`df -h /scratch`).
- Caminhos fixos no código em vez de argumento/variável de ambiente.
