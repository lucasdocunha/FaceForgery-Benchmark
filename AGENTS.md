# AGENTS.md

Instruções para qualquer agente (Claude Code, Gemini, Codex etc.) que trabalhe neste repositório.

## Registro de experimentos no Linear

Todo experimento do artigo é registrado no Linear. Assim quem não roda no servidor sabe o que foi rodado, como, e o que deu, sem precisar perguntar a quem rodou.

### Onde

- Workspace `tcc-grupo-06` (<https://linear.app/tcc-grupo-06>), time `TCC`. Nunca escreva em outro workspace do Linear.
- A chave da API fica no `.env` da raiz do repositório, como `LINEAR_API_KEY` (o `.env` é ignorado pelo git). Nunca imprima a chave, não a copie para arquivos versionados e não a envie a nenhum destino além de `api.linear.app` e `mcp.linear.app`.
- Se houver um servidor MCP do Linear conectado, use-o só depois de confirmar com `get_workspace` que ele aponta para `tcc-grupo-06`. Se ele apontar para outro workspace, ou se não houver MCP, use a API GraphQL com a chave do `.env` (exemplo em `docs/linear-experimentos.md`).
- Se o Linear estiver inacessível, não invente nada: mostre ao usuário o texto pronto para ele colar.

### Modelo

- 1 issue = 1 experimento (uma hipótese). Correção de bug sem mudar a hipótese vira uma nova rodada ("Rodada N") na mesma issue. Variante nova vira outra issue, marcada como relacionada.
- Projetos são as linhas de pesquisa: Desempenho no Teste, Robustez no Test-D, Generalização cross-dataset, Análise de erros. Toda issue fica em um projeto.
- "Servidor" é o CISIA ou a workstation do Lucas Cunha; o fluxo é o mesmo nos dois.

| Status | Significado | Quem move |
|---|---|---|
| Backlog | ideia ou hipótese | qualquer um |
| A fazer | aceita, não começada | qualquer um |
| Implementando | código sendo escrito | autor |
| Pronto para rodar | spec completa (modelo A), commit fixado, validação local feita | autor |
| Rodando | job submetido (modelo B) | quem roda |
| Em análise | resultados postados (modelo C), falta conclusão | quem roda |
| Concluído | conclusão escrita (modelo D) | quem conclui |
| Cancelado | abandonado; explique o motivo em um comentário | qualquer um |

Etiquetas: `tipo` (experimento, análise, infra), `escopo` (piloto = dataset mínimo ou teste local; benchmark = dataset completo) e `resultado` (positivo, negativo, inconclusivo; obrigatória em Concluído).

### Quando escrever

Os modelos A a D estão em `docs/linear-experimentos.md`. Leia esse arquivo antes de escrever no Linear.

1. Autor, com o código pronto: descrição da issue no modelo A, status Pronto para rodar, atribuída a quem vai rodar.
2. Quem roda, ao submeter: comentário no modelo B, status Rodando.
3. Quem roda, ao terminar ou falhar: comentário no modelo C, status Em análise. Falha também é resultado e deve ser registrada.
4. Quem conclui: comentário no modelo D, etiqueta `resultado`, status Concluído, issues de continuação criadas e ligadas.

Se o usuário pedir algo como "posta os resultados da TCC-12", encontre as saídas daquela rodada (log com o ID da issue no nome, `JOB.json`, saída da suíte) e siga o passo 3.

### Regras de conteúdo

- Números só copiados de arquivos de saída (`JOB.json`, `suite_metrics.json`, `metrics.json`/`metrics.csv` de cada alvo, saída do `aggregate-seeds`, `.out`), sempre com o caminho do arquivo. Nunca de memória ou de um resumo. O que não estiver em arquivo é "não registrado".
- Separe **Fatos** de **Interpretação**.
- Registre todo desvio da spec: commit, seeds, épocas, argumentos, dataset, máquina.
- Coloque o ID da issue no nome do job (`sbatch --job-name=TCC-12-cae ...`) ou no nome do log em rodadas locais (`logs/TCC-12-cae_<data>.log`), e nas mensagens de commit.
- Antes de Pronto para rodar no CISIA, rode a checagem de `skill-hpc/SKILL.md` e registre o veredito.
- Nunca envie ao Linear imagens, rostos, `predictions.csv` ou outros arquivos por amostra, pesos ou segredos. Dados brutos vão como texto: `suite_metrics.json`, trechos do `JOB.json` e do `.out`/`.err`.
- Escreva em português. Nomes de métricas, datasets, arquivos e código ficam como estão.
- Número que vai para o artigo precisa vir de um comentário de resultados (modelo C).

### Referências do projeto

- Cluster CISIA: `skill-hpc/SKILL.md` e `docs/cisia-hpc.md`.
- Experimentos novos: `research/experimental_extensions/SERVER_RUNBOOK.md` (comandos e artefatos), `experiment_matrix.yaml` (nome, config, comandos e saídas esperadas de cada experimento) e `REPORT.md` (pilotos e baselines).
- Seeds canônicas: 42, 123, 2024, 7, 2025.
- Alvos de avaliação: Test (MFFI limpo), Test-D (MFFI com 14 corrupções), DF-40 (por gerador e por paradigma) e Celeb-DF v2 (AUC por vídeo é a principal).
- Baselines atuais: tabela "Existing evidence and comparison targets" em `research/experimental_extensions/REPORT.md`.
