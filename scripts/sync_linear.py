"""Sincroniza os resultados experimentais no Linear (TCC-15 e nova issue de Cross-Dataset)."""

import json
import re
import urllib.request
import pathlib

KEY = re.search(r"^LINEAR_API_KEY=(.*)$", pathlib.Path(".env").read_text(), re.M).group(1).strip().strip("\"'")

def gql(query, **variables):
    req = urllib.request.Request(
        "https://api.linear.app/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json", "Authorization": KEY},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
    if "errors" in result:
        raise RuntimeError(result["errors"])
    return result["data"]

def main():
    org = gql("query { organization { urlKey } }")["organization"]["urlKey"]
    assert org == "tcc-grupo-06", f"Workspace incorreto: {org}"
    print(f"Conectado ao workspace: {org}")

    # Obter IDs necessários do Linear
    team_data = gql("""
    query {
        team(id: "TCC") {
            id
            states { nodes { id name } }
            projects { nodes { id name } }
            labels { nodes { id name } }
        }
    }
    """)
    team_id = team_data["team"]["id"]
    states = {s["name"]: s["id"] for s in team_data["team"]["states"]["nodes"]}
    projects = {p["name"]: p["id"] for p in team_data["team"]["projects"]["nodes"]}
    labels = {l["name"]: l["id"] for l in team_data["team"]["labels"]["nodes"]}

    # Obter ID da issue TCC-15
    tcc15 = gql('query($id: String!) { issue(id: $id) { id identifier title } }', id="TCC-15")["issue"]
    print(f"Localizada: {tcc15['identifier']} - {tcc15['title']}")

    # 1. Postar Comentário Rodada 2 na TCC-15
    tcc15_comment = """### Rodada 2: resultados
**Estado:** concluída com sucesso (Slurm Jobs 143 na GPU 0 e 144 na GPU 1).
Duração: ~50 minutos somando todos os 17 modelos (throughput médio: 350-430 imgs/s por GPU). Commit: `537a45c`.

**Fatos**
Inferência exaustiva de todas as 650.549 imagens falsas do DF-40 para os 17 modelos campeões (totalizando 11.059.333 predições):

| Rank | Modelo | Paradigma Predominante | Total Acertos | Taxa Geral (%) | HeyGen (Avatares) (%) | Fonte |
|---|---|---|---|---|---|---|
| 🥇 | DINO (FFT Multi-Freq) | Frequência (6ch) | 607.639 | 93,40% | 97,32% | `tables/df40_all_fakes_models/df40_all_fakes_dino_concat_freq.csv` |
| 🥈 | Xception (SRM) | Resíduo SRM (3ch) | 600.534 | 92,31% | 87,85% | `tables/df40_all_fakes_models/df40_all_fakes_xception_srm.csv` |
| 🥉 | DINO (SRM Only) | Resíduo SRM (3ch) | 595.435 | 91,53% | 86,11% | `tables/df40_all_fakes_models/df40_all_fakes_dino_srm_only.csv` |
| 4 | Xception (SRM Only) | Resíduo SRM (3ch) | 593.449 | 91,22% | 96,14% | `tables/df40_all_fakes_models/df40_all_fakes_xception_srm_only.csv` |
| 5 | DINO (SRM) | Resíduo SRM (3ch) | 563.981 | 86,69% | 32,96% | `tables/df40_all_fakes_accuracy_breakdown.csv` |
| 6 | ResNet-18 (FFT Multi-Freq) | Frequência (6ch) | 555.915 | 85,45% | 94,95% | `tables/df40_all_fakes_models/df40_all_fakes_resnet_concat_freq.csv` |
| 7 | ResNet-18 (SRM Only) | Resíduo SRM (3ch) | 550.469 | 84,62% | 61,00% | `tables/df40_all_fakes_models/df40_all_fakes_resnet_srm_only.csv` |
| 8 | ViT (SRM) | Resíduo SRM (3ch) | 543.510 | 83,55% | 41,50% | `tables/df40_all_fakes_models/df40_all_fakes_vit_srm.csv` |
| 9 | DINO (RGB Robusto) | RGB (3ch) | 541.988 | 83,31% | 30,72% | `tables/df40_all_fakes_models/df40_all_fakes_dino_rgb_robust.csv` |
| 10 | MobileNet (SRM) | Resíduo SRM (3ch) | 534.782 | 82,20% | 29,47% | `tables/df40_all_fakes_models/df40_all_fakes_mobilenet_srm.csv` |
| 11 | MobileNet (RGB Robusto) | RGB (3ch) | 525.906 | 80,84% | 31,21% | `tables/df40_all_fakes_models/df40_all_fakes_mobilenet_rgb_robust.csv` |
| 12 | DINO (FFT Concat) | Frequência (7ch) | 525.082 | 80,71% | 69,66% | `tables/df40_all_fakes_models/df40_all_fakes_dino_concat.csv` |
| 13 | CLIP (SRM) | Resíduo SRM (3ch) | 501.598 | 77,10% | 31,40% | `tables/df40_all_fakes_models/df40_all_fakes_clip_srm.csv` |
| 14 | ResNet-18 (SRM) | Resíduo SRM (3ch) | 486.894 | 74,84% | 22,12% | `tables/df40_all_fakes_models/df40_all_fakes_resnet_srm.csv` |
| 15 | ResNet-18 (RGB Robusto) | RGB (3ch) | 481.444 | 74,01% | 17,63% | `tables/df40_all_fakes_models/df40_all_fakes_resnet_rgb_robust.csv` |
| 16 | ResNet-18 (FFT Concat) | Frequência (7ch) | 479.265 | 73,67% | 70,84% | `tables/df40_all_fakes_models/df40_all_fakes_resnet_concat.csv` |
| 17 | CLIP (RGB Robusto) | RGB (3ch) | 227.414 | 34,96% | 13,52% | `tables/df40_all_fakes_models/df40_all_fakes_clip_rgb_robust.csv` |

Acurácia consolidada por paradigma (fonte: `tables/df40_all_fakes_progress_matrix_by_paradigm.csv`):
- **Difusão:** DINO FFT Multi-Freq (91,34%), Xception SRM (90,28%).
- **Síntese GAN:** DINO FFT Multi-Freq (96,08%), DINO SRM Only (92,25%).
- **Troca Facial:** DINO SRM Only (96,23%), DINO FFT Multi-Freq (95,02%).
- **Edição / T2I:** Xception SRM Only (98,05%), Xception SRM (96,89%).
- **Avatares Comerciais (HeyGen):** DINO FFT Multi-Freq (97,32%), Xception SRM Only (96,14%), ResNet-18 FFT Multi-Freq (94,95%).

**Interpretação**
1. O domínio de frequência e ruído puro supera amplamente o RGB na detecção de fakes modernos: DINO FFT Multi-Freq (93,40%) e Xception SRM (92,31%) superam o DINO RGB Robusto (83,31%) em mais de +9 pp no geral.
2. No caso crítico do HeyGen (avatares comerciais com compressão de vídeo H.264), redes baseadas em RGB falham catastroficamente (13% a 31% de acerto), enquanto o ruído SRM puro e decomposição FFT atingem até 97,32%.
3. Modelos ViT sem pré-processamento de frequência ou sem ruído residual sofrem expressivamente com compressões modernas."""

    gql("mutation($input: CommentCreateInput!) { commentCreate(input: $input) { success } }",
        input={"issueId": tcc15["id"], "body": tcc15_comment})
    print("✅ Comentário da Rodada 2 adicionado com sucesso em TCC-15!")

    # 2. Criar Nova Issue para o Benchmark Reverso / Cross-Dataset
    issue_desc = """## Hipótese
Treinar modelos forenses utilizando o conjunto diversificado do DF-40 (composto por 24 métodos modernos abrangendo Difusão, GANs, Face Swapping e Avatares Comerciais) permite testar a hipótese de assimetria no cross-dataset: investigar se detectores treinados em sínteses modernas generalizam de volta para manipulações arcaicas (MFFI Test) e corrupções de canal (MFFI Test-D e Celeb-DF v2).

## Linha de pesquisa
Projeto: Generalização cross-dataset.
Ligação com TCC-7 (investigação de queda de cross-dataset), TCC-8 (SRM canais extras), TCC-15 (auditoria de erros DF-40) e TCC-17 (SRM puro).

## Código
- Branch: `feat/srm-only` | Commit: `3d23b95` | PR: nenhum
- Arquivos principais:
  - `scripts/prepare_df40_training_splits.py` (splits 16k train / 4k val sem vazamento do test.csv)
  - `scripts/train_and_eval_df40_cross.py` (treino e avaliação nos 4 alvos)
  - `scripts/slurm_df40_cross_dino.sh`
  - `scripts/slurm_df40_cross_resnet.sh`
  - `tables/df40_cross_dataset_comparison.csv`

## Como rodar
- Onde: Workstation local (`sicret2`) com 2x NVIDIA RTX 3090 24GB via Slurm
- Comandos:
  - `sbatch scripts/slurm_df40_cross_dino.sh`
  - `sbatch scripts/slurm_df40_cross_resnet.sh`
- Seeds: 42 (piloto canônico comparativo)

## Saídas esperadas
- Logs: `logs/TCC-df40-cross_dino_154.out`, `logs/TCC-df40-cross_resnet_155.out`
- Checkpoints: `models/df40_trained/dino_srm_seed_42/weights/best.pth`, `models/df40_trained/resnet_srm_seed_42/weights/best.pth`
- Arquivos de métricas: `models/df40_trained/*/cross_dataset_results.json` e `tables/df40_cross_dataset_comparison.csv`

## Critério de sucesso
Avaliar a retenção de generalização cruzada reversa: verificar se a AUC no MFFI Test atinge >= 70% e no Celeb-DF v2 >= 70% quando treinados puramente no DF-40.
"""

    created_issue = gql("""
    mutation($input: IssueCreateInput!) {
        issueCreate(input: $input) {
            success
            issue { id identifier title url }
        }
    }
    """, input={
        "teamId": team_id,
        "title": "Benchmark Reverso: Treinamento com DF-40 e Avaliação Cruzada no MFFI e Celeb-DF v2",
        "description": issue_desc,
        "projectId": projects["Generalização cross-dataset"],
        "stateId": states["Em análise"],
        "labelIds": [labels["experimento"], labels["benchmark"]],
    })["issueCreate"]["issue"]

    print(f"✅ Nova issue criada: {created_issue['identifier']} - {created_issue['title']} ({created_issue['url']})")

    # Relacionar a nova issue com TCC-15
    gql("""
    mutation($input: IssueRelationCreateInput!) {
        issueRelationCreate(input: $input) {
            success
        }
    }
    """, input={
        "issueId": created_issue["id"],
        "relatedIssueId": tcc15["id"],
        "type": "related"
    })
    print(f"✅ Relação criada entre {created_issue['identifier']} e TCC-15!")

    # 3. Postar Comentário de Resultados (Modelo C) na Nova Issue
    cross_results_comment = """### Rodada 1: resultados
**Estado:** concluída com sucesso (Slurm Jobs 151, 153, 154 e 155 nas GPUs 0 e 1).
Duração: ~12 minutos de treino por modelo + ~13 minutos de avaliação exaustiva em todos os 4 alvos. Commit: `3d23b95`.

**Fatos**

Tabela comparativa direta de desempenho: Modelo treinado na MFFI vs. Modelo treinado no DF-40 (fonte: `tables/df40_cross_dataset_comparison.csv`):

| Modelo | Dataset de Treino | MFFI Test (Limpo)<br>AUC / Acc | MFFI Test-D (Corrompido)<br>AUC / Acc | DF-40 Test (Intra/Cross)<br>AUC / Acc | Celeb-DF v2 (Vídeo)<br>AUC / Acc | Celeb-DF v2 (Frame)<br>AUC / Acc |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **DINO (SRM)** | **MFFI** | **91,26%** / **80,83%** | **84,72%** / **72,61%** | 79,52% / 75,33% | **83,19%** / **72,59%** | **77,88%** / **72,75%** |
| **DINO (SRM)** | **DF-40** | 72,60% / 66,08% | 63,08% / 59,00% | **99,92%** / **98,92%** | 70,04% / 62,36% | 68,34% / 68,49% |
|---|---|---|---|---|---|---|
| **ResNet-18 (SRM)** | **MFFI** | **83,72%** / **75,40%** | **75,42%** / **68,52%** | 68,04% / 65,00% | **65,75%** / 67,00% | **61,20%** / 62,50% |
| **ResNet-18 (SRM)** | **DF-40** | 56,42% / 54,81% | 52,19% / 52,08% | **99,28%** / **94,78%** | 65,44% / **69,50%** | **62,62%** / **67,16%** |

Fontes detalhadas:
- DINO (SRM) no DF-40: `models/df40_trained/dino_srm_seed_42/cross_dataset_results.json`
- ResNet-18 (SRM) no DF-40: `models/df40_trained/resnet_srm_seed_42/cross_dataset_results.json`
- DINO (SRM) na MFFI: `models-tcc/dino/srm/finetune_robust/seed_42/results/metrics_*.csv`
- ResNet-18 (SRM) na MFFI: `tables/campaign_srm_summary_5seeds.csv`

**Critério de sucesso:** Atingido parcialmente.
- O DINO (SRM) treinado no DF-40 atingiu 72,60% de AUC no MFFI Test limpo e 70,04% de Video AUC no Celeb-DF v2 (acima da meta de 70%).
- Já o ResNet-18 (SRM) não sustentou generalização cruzada reversa satisfatória (56,42% no MFFI Test).

**Interpretação**
1. **Domínio quase perfeito no DF-40:** Treinar diretamente no DF-40 com SRM eleva a acurácia para 98,92% (AUC de 99,92% no DINO), demonstrando que ruídos residuais de alta frequência aprendem eficazmente as assinaturas dos 24 métodos contemporâneos de difusão e GANs.
2. **Assimetria de generalização cruzada:** Modelos treinados em conjuntos focados em Difusão e GANs modernas (DF-40) têm menor capacidade de generalização reversa para manipulações clássicas de warping facial (FF++/MFFI) do que o inverso. A representação da MFFI fornece uma base de detecção mais generalista para manipulações de reenactment e swap (alcançando 79,52% no DF-40 e 83,19% no Celeb-DF v2).
3. **Resiliência do DINO frente à CNN rasa:** Enquanto o ResNet-18 despenca para nível próximo ao acaso (56% AUC no MFFI Test), o DINO sustenta 72,60% AUC no MFFI Test e 70,04% de AUC no Celeb-DF v2, demonstrando a superioridade de representações baseadas em modelos fundacionais com atenção espacial.

**Artefatos no servidor:**
- Checkpoints: `models/df40_trained/dino_srm_seed_42/weights/best.pth`, `models/df40_trained/resnet_srm_seed_42/weights/best.pth`
- Tabela consolidada: `tables/df40_cross_dataset_comparison.csv`
- Resultados em JSON: `models/df40_trained/*/cross_dataset_results.json`"""

    gql("mutation($input: CommentCreateInput!) { commentCreate(input: $input) { success } }",
        input={"issueId": created_issue["id"], "body": cross_results_comment})
    print(f"✅ Comentário da Rodada 1 adicionado com sucesso em {created_issue['identifier']}!")

if __name__ == "__main__":
    main()
