---
name: cisia-hpc
description: Checagem de prontidão de qualquer projeto que vai rodar no cluster CISIA (SLURM, 2 nós × 4 H100, OOD/Job Composer). Confere código e script .sh contra as regras do cluster — ambiente conda, recursos #SBATCH, caminhos, armazenamento NFS × disco local, datasets, modelos, logs e serviços dentro do job (ex.: Ollama) — e devolve um relatório ✅/❌/⚠️ com o que mudar. Use ao escrever, revisar ou preparar código/sbatch para o CISIA, ou quando o usuário mencionar CISIA, SLURM, sbatch, srun, cluster, Job Composer, partição gpu/shared.
---

# CISIA — checagem de prontidão

Objetivo: antes de o usuário submeter, dizer **se o job vai rodar no CISIA e seguir o padrão**, lendo o código e o `.sh`. Checagem **estática**: não executa nada, não gera job de teste, não monta container.

**Não edite o código do projeto.** A skill valida e direciona: cada problema sai no relatório com *onde* (`arquivo:linha`) e *o que mudar*. O `.sh` é a exceção parcial: se não existir ou estiver fora do padrão, **sugira** o `.sh` completo no relatório, a partir de [template.sh](template.sh) — escreva o arquivo só se o usuário pedir.

## Fluxo

1. Leia [cluster.md](cluster.md) (fatos do cluster — são a referência de tudo abaixo).
2. Localize o entrypoint, o `.sh` de submissão (se houver) e tudo que o código lê/grava.
3. Percorra cada checagem, lendo o arquivo dela:

| Checagem | Arquivo | Pergunta |
|---|---|---|
| Ambiente | [ambiente.md](ambiente.md) | O env conda/módulos/containers que o job usa existe nos dois nós e tem o que o código importa? |
| Recursos | [recursos.md](recursos.md) | O `#SBATCH` é válido, cabe no teto e é proporcional ao que o código usa? |
| Armazenamento | [armazenamento.md](armazenamento.md) | Cada leitura/escrita vai pro lugar certo (NFS × disco local), sem estourar arquivos por pasta? |
| Datasets | [datasets.md](datasets.md) | Lê de `/datasets` sem escrever lá; dataset novo segue o layout? |
| Modelos | [modelos.md](modelos.md) | Carrega do que o cluster já tem, salva no lugar e formato padrão? |
| Logs | [logs.md](logs.md) | O `.out` terá cabeçalho, progresso, rodapé e métricas? |
| Serviços | [servicos.md](servicos.md) | Só se o job sobe servidor (Ollama, vLLM, API): porta/PID isolados por job? |

4. Emita o relatório.

## Relatório

```
## Prontidão CISIA — <projeto> (<entrypoint>)

Veredito: PRONTO | PRONTO COM AVISOS | NÃO RODA

| # | Checagem | Status | Achado | Onde | O que mudar |
|---|---|---|---|---|---|
| 1 | Ambiente | ✅/❌/⚠️ | ... | arquivo:linha | ... |
...

### .sh sugerido          (só se faltou ou está fora do padrão)
<script completo>

### Comandos pra rodar no Shell Access antes de submeter   (só se algo não dá pra ver daqui)
<ex.: conda list -n <env> | grep -E 'torch|transformers'; mkdir -p logs saidas>
```

- ❌ **bloqueia**: o job falha ou quebra regra do cluster (caminho inexistente, env local a um nó, `gpu:` na `shared`, cache no NFS, `logs/` inexistente, store de modelo próprio no NFS).
- ⚠️ **aviso**: roda, mas foge do padrão (log sem progresso, recurso superdimensionado, saída fora de `saidas/`).
- O que depende de estado do cluster que você não vê (env existe? pasta existe?) vira item de "Comandos pra rodar", nunca ✅ presumido.
