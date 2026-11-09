# Checagem: serviços dentro do job

Só se aplica quando o job sobe um servidor (Ollama, vLLM, TGI, API local, banco efêmero) e o código fala com ele. Os nós são compartilhados: dois jobs (seus ou de colegas) podem estar no mesmo nó ao mesmo tempo.

## Regras

- **Porta dinâmica**: nunca porta fixa. Escolher uma livre no início do job (ex.: `python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])'`) e passá-la ao servidor e ao código.
- **Isolamento por `SLURM_JOB_ID`**: arquivo de config/PID do serviço nomeado com o job id (ex.: `$CACHE_DIR/servico_${SLURM_JOB_ID}.pid`).
- **Encerrar só o próprio**: `kill "$PID"` do PID que o job iniciou, no `trap EXIT`. Nunca `pkill -u $USER`, `killall`, `pkill ollama`.
- **Esperar ficar pronto**: loop de health-check (curl na porta) com timeout antes de o código começar; falhou → `exit 1` com mensagem.
- **GPU**: `CUDA_VISIBLE_DEVICES` dentro do job é índice local (`0`...), nunca o índice global do nó.
- Dados/blobs do serviço: store compartilhado só leitura ou cache local do job, nunca store próprio no NFS.

## Exemplo: Ollama

```bash
module load ollama/0.20.7    # exige #!/bin/bash -l; define OLLAMA_MODELS=/datasets/ai_models/ollama e OLLAMA_HOST=127.0.0.1:11434
PORTA=$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:${PORTA}"   # sobrescreve a porta fixa do módulo — senão dois jobs no mesmo nó colidem
export OLLAMA_NOPRUNE=1                   # o store compartilhado é só leitura
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null; limpar' EXIT
for _ in $(seq 60); do curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null && break; sleep 2; done
curl -sf "http://${OLLAMA_HOST}/api/tags" | grep -q '"<modelo>"' || { echo "modelo <modelo> ausente em $OLLAMA_MODELS"; exit 1; }
```

- Modelo que não está em `/datasets/ai_models/ollama`: store por job — `export OLLAMA_MODELS="$CACHE_DIR/ollama"` antes do `serve`, e `ollama pull`/`ollama create` (adapter próprio) lá dentro; custa rebaixar a cada job. Se o modelo vai ser usado por muito tempo, pedir aos admins que o incluam no store compartilhado.
- ❌ `OLLAMA_MODELS` em `$HOME`/`/projects` (EIO no NFS); ❌ `OLLAMA_HOST` na porta fixa do módulo; ❌ `pkill ollama`.
