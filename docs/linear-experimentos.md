# Linear: modelos e uso da API

Complemento do `AGENTS.md`, que define onde e quando escrever. Aqui estão os modelos de texto e um exemplo de acesso ao Linear sem MCP.

Em todos os modelos: números só copiados de arquivos, com o caminho; o que não estiver em arquivo é "não registrado"; apague as linhas que não se aplicam, em vez de deixá-las vazias.

## Modelo A: descrição da issue (autor, ao mover para Pronto para rodar)

```markdown
## Hipótese
O que esperamos observar e por quê, em uma ou duas frases.

## Linha de pesquisa
Projeto e ligação com resultados anteriores (links para as issues).

## Código
- Branch: `...` | Commit: `<sha completo>` | PR: <link ou "nenhum">
- Arquivos principais: `...`
- Nome em `experiment_matrix.yaml`: `<name>` (se houver)

## Como rodar
- Onde: CISIA | workstation
- Comando exato (com o ID da issue no nome do job ou do log):
  `sbatch --job-name=TCC-<n>-<slug> scripts/...`
- Config: `configs/...` | Seeds: 42, 123, 2024, 7, 2025
- Recursos e tempo estimado: GPU, memória, horas
- Avaliação nos alvos, se o script não fizer: comando da suíte e do `aggregate-seeds`

## Saídas esperadas
- Log: `logs/TCC-<n>-<slug>_<jobid>.out`
- Run dir, saída da suíte, `JOB.json`: caminhos

## Critério de sucesso (definido antes de rodar)
Ex.: AUC por vídeo no Celeb-DF v2 acima do baseline de 0.7739 sem o Test cair abaixo de 0.93.

## Validação local
- Testes: comando e resultado
- Piloto em dataset mínimo: números com fonte, ou "não feito"
- Checagem `skill-hpc`: PRONTO | PRONTO COM AVISOS | NÃO RODA

## Resultado
Preenchido na conclusão: uma frase e o link para o comentário de resultados.
```

## Modelo B: comentário ao submeter (quem roda)

```markdown
### Rodada <N>: submetida
- Data e hora:
- Onde: CISIA (nó) | workstation
- Job ID(s):
- Commit rodado: `<sha>` (se diferente do modelo A, por quê)
- Comando exato:
- Desvios da spec: nenhum | lista
```

## Modelo C: comentário de resultados (quem roda, ao terminar ou falhar)

Seja enxuto: o comentário deve ser lido em poucos minutos. Fatos são a tabela principal e no máximo cinco tópicos; detalhes de protocolo e caminhos completos ficam nos arquivos citados. Antes de postar, confira cada número e cada afirmação contra o arquivo de origem.

```markdown
### Rodada <N>: resultados
**Estado:** concluída | falhou | parcial. Código de saída e duração (fonte: `.out` ou `sacct`).

**Fatos**

| Alvo | Seed | AUC [IC 95%] | AP | EER | Acurácia balanceada | Fonte |
|---|---|---|---|---|---|---|
| Test | 42 | | | | | `<caminho>/test/metrics.json` |
| ... | | | | | | |
| Test | média ± DP | | | | | saída do `aggregate-seeds` |
| Test | baseline | | | | | `research/experimental_extensions/REPORT.md` |

- Test-D menos Test (pareado): valor [IC] (fonte: `suite_metrics.json`)
- DF-40, três piores geradores: valores (fonte: `per_generator.csv`)
- Celeb-DF v2: AUC por vídeo e por frame
- Erros e avisos: trecho do `.err`/`.out`

**Critério de sucesso:** atingido | não atingido | não avaliável, com a comparação numérica.

**Interpretação** (opinião de quem postou)

**Artefatos no servidor:** run dir, saída da suíte, modelos (caminhos).

**Dados brutos**
Só as métricas agregadas (no máximo umas 40 linhas) de `suite_metrics.json` ou do `JOB.json`, em um bloco de código, com o caminho do arquivo completo.
```

## Modelo D: conclusão (quem conclui)

```markdown
### Conclusão
- Resultado: positivo | negativo | inconclusivo, em uma frase.
- O que aprendemos:
- Próximos passos: links para as issues novas.
```

## Acesso sem MCP (API GraphQL)

Funciona em qualquer agente com Python. A chave é lida do `.env` dentro do próprio script e nunca é impressa.

```python
import json, re, urllib.request, pathlib

KEY = re.search(r"^LINEAR_API_KEY=(.*)$", pathlib.Path(".env").read_text(), re.M).group(1).strip().strip("\"'")

def gql(query, **variables):
    request = urllib.request.Request(
        "https://api.linear.app/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json", "Authorization": KEY},
    )
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    if "errors" in result:
        raise RuntimeError(result["errors"])
    return result["data"]

# Confirme o workspace antes de escrever.
assert gql("query { organization { urlKey } }")["organization"]["urlKey"] == "tcc-grupo-06"

issue = gql('query($id: String!) { issue(id: $id) { id title description state { name } comments { nodes { body } } } }', id="TCC-12")["issue"]
states = {s["name"]: s["id"] for s in gql('query { team(id: "TCC") { states { nodes { id name } } } }')["team"]["states"]["nodes"]}

gql("mutation($input: CommentCreateInput!) { commentCreate(input: $input) { success } }",
    input={"issueId": issue["id"], "body": "### Rodada 1: resultados\n..."})
gql("mutation($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { success } }",
    id=issue["id"], input={"stateId": states["Em análise"]})
```

Outras mutações úteis: `issueCreate` (com `teamId`, `projectId`, `labelIds`, `stateId`), `issueRelationCreate` (tipo `related`) e `issueUpdate` com `labelIds`. Os IDs de projetos e etiquetas saem de `projects { nodes { id name } }` e `issueLabels { nodes { id name } }`.
