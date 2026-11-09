# Roteiro de apresentação — pt-BR

Médias arredondadas do estudo anterior, reproduzidas na seção V-A da edição de seis páginas. Não são uma reavaliação dos pesos fine-tuned atuais. As diferenças são subtrações das médias reportadas; não são testes de significância.

## 01. Detecção de falsificações faciais

Apresente primeiro os resultados do benchmark anterior. Em seguida, explique os três protocolos implementados para investigar as decisões dos detectores. Esta apresentação não contém heatmaps novos nem resultados inferidos como se tivessem sido medidos. Autores mantidos conforme o manuscrito fornecido; o material requer revisão da equipe.

Fontes: A · Estudo anterior; B · seção V-A.

## 02. O que está sendo comparado

O MFFI contém, no estudo anterior, 524.429 imagens de treino, 147.363 de validação e 181.947 de teste, além do teste degradado correspondente. Os resultados são médias de três sementes. Cinco famílias foram treinadas do zero; a configuração DINOv3 do estudo anterior tinha o backbone congelado. Não confundir esses modelos com o release fine-tuned usado pela extensão. ROC-AUC resume a ordenação entre classes ao variar o limiar; não é a porcentagem de acertos em um limiar específico.

Fontes: A · seções III–IV; B · seção III.

## 03. A degradação muda o ranking

Todos os pontos são médias RGB do estudo anterior. Círculo azul indica teste limpo; quadrado verde indica teste degradado. O eixo começa em 0,50 e termina em 1,00, de forma explícita: é um gráfico de posições, não de barras. Xception lidera o limpo, enquanto DINOv3 congelado lidera o degradado entre as seis configurações mostradas. A figura não adiciona intervalos de confiança nem prova superioridade estatística. DINOv3 e os demais modelos diferem também no regime de pré-treinamento.

Fontes: A · Estudo anterior; B · seção V-A.

## 04. Pequena queda, sozinha, não é robustez

A queda absoluta do ViT é menor, mas seu ponto de partida já é baixo: 0,624 no limpo e 0,608 no degradado. DINOv3 cai mais, 0,083, porém mantém AUC degradada maior, 0,726. Portanto, classificar robustez apenas pela menor queda pode favorecer um modelo pouco discriminativo. Estes são contrastes descritivos; não se isola a arquitetura do pré-treinamento.

Fontes: A · Estudo anterior; B · seção V-A.

## 05. Frequência: complemento, não substituição

Três comparações descritas no artigo anterior: Xception RGB+magnitude melhora de 0,609 para 0,650; ResNet híbrido de 0,635 para 0,648; MobileNet RGB+magnitude de 0,594 para 0,612 no teste degradado. Os ganhos são 0,041, 0,013 e 0,018 unidades de AUC, calculados das médias arredondadas. São exemplos reportados, não todos os modos nem uma seleção feita em validação nesta apresentação. Não chamar AUC de porcentagem de acerto nem os ganhos de prova causal.

Fontes: A · Estudo anterior; B · seção V-A.

## 06. O que os resultados permitem concluir

Separar observação de mecanismo: o ranking muda e existem ganhos em entradas híbridas selecionadas no artigo. Esses resultados motivam investigar evidência local ou espectral, mas não demonstram que uma rede identificou um tipo específico de artefato. Tampouco isolam efeitos de arquitetura, pré-treinamento, identidade ou fonte. As análises de explicabilidade entram para formular e examinar essas hipóteses.

Fontes: A · Estudo anterior; B · seção V-A.

## 07. Três perguntas para explicar as decisões

A partir daqui, apresenta-se o protocolo implementado, não resultados recém-observados. A primeira tarefa controla as imagens; a segunda controla a comparação RGB–frequência; a terceira define a população de falhas compartilhadas. Todas preservam rastreabilidade, limiares congelados e limitações de interpretação.

Fontes: B · seção IV; C · guia de execução.

## 08. Tarefa 1 · As mesmas 64 imagens

A classe positiva é falsa. O modelo de referência é ResNet/RGB/seed 42; a semente de amostragem é 42. VP: falsa prevista falsa; FN: falsa prevista real; VN: real prevista real; FP: real prevista falsa. São 16 casos sem reposição em cada estrato. As mesmas identidades e ordem são reutilizadas; outros modelos mantêm seus próprios desfechos. Essa amostra super-representa erros, portanto não estima acurácia populacional nem prevalência de falhas.

Fontes: B · tarefa 1; C · configuração.

## 09. Tarefa 2 · Comparar no domínio correto

A comparação exige checkpoints separadamente treinados, da mesma família, regime e semente, em RGB e frequência. A seleção usa apenas validação e um piso de competência, não o ranking do teste. A implementação compara as mesmas 64 imagens, com atribuições em pixels sobre RGB e em bins sobre a entrada de frequência. Este slide é um esquema textual, sem heatmaps simulados. Um coeficiente global de Fourier não corresponde a uma posição facial; não calcular sobreposição espacial RGB–espectro como se fossem as mesmas coordenadas.

Fontes: B · tarefa 2; C · seleção e renderização.

## 10. Tarefa 3 · Falhas do conjunto declarado

A interseção exige todos os modelos no mesmo conjunto de imagens, com IDs únicos, rótulos consistentes e predições completas. A população primária usa seis checkpoints RGB seed 42; a análise de sensibilidade usa os 18 checkpoints de três sementes. São mantidas todas as falhas compartilhadas, sem teto oculto. Os controles são pareados por rótulo e não pertencem à interseção; podem falhar em parte dos modelos. Iluminação e oclusão exigem anotação e comparação; rótulo pareado não controla identidade, fonte ou compressão. Não inferir demografia automaticamente.

Fontes: B · tarefa 3; C · controles e revisão humana.

## 11. 12 métodos, perguntas diferentes

Métodos agrupados pela forma predominante de análise: seis por gradientes/localização, cinco por perturbação/surrogates e um de atenção. Kernel SHAP e LIME produzem contribuições de grupos sob perturbação; Grad-CAM localiza em uma camada; IG é relativo a baseline. Attention Rollout é agnóstico à classe e não é equivalente à explicação por gradiente da margem falso-real. Nem todos os métodos são aplicáveis a todos os modelos. O número de métodos não corresponde a confirmações independentes.

Fontes: B · tabela de métodos; D–G · base metodológica.

## 12. O que está pronto e o que falta medir

A parte implementada inclui amostragem e seleção reproduzíveis, atribuições, exportação, hashes, verificação das predições, retomada por partições e controles numéricos. As experiências de imagem ainda precisam ser executadas para produzir as galerias e a análise de falhas. Nesta apresentação não se treina modelo, não se usa GPU e não se produz nova AUC. Testes sintéticos de software e compilação não são evidência de fidelidade das explicações em dados reais.

Fontes: B · seção VI; C · execução e cobertura.

## 13. Três mensagens para levar

Encerrar com três mensagens: avaliar no limpo não substitui avaliar sob degradação; sinais espectrais podem complementar RGB mas não revelam automaticamente o mecanismo; explicar exige controlar a amostra, o alvo, as coordenadas e os métodos. As tarefas implementadas permitem investigar essas questões sem antecipar achados. Os materiais completos e fontes estão na pasta presentation e no notebook.

Fontes: A–C · resultados anteriores e extensão.

## 14. Apoio · Tabela completa dos resultados RGB

Tabela de consulta com todas as seis médias RGB e suas quedas. O valor na última coluna é limpo menos degradado, com Decimal para preservar a subtração das médias arredondadas. Não misturar esses números com o release atual fine-tuned. Os desvios-padrão do estudo original não são reconstruídos a partir de médias. Fontes A e B indicam a origem; dados.json permite editar e verificar o material.

Fontes: A · Estudo anterior; B · seção V-A.

## 15. Fontes e material de apoio

Os links de documentos do repositório estão fixados no commit usado para construir a apresentação. A referência A contém o estudo original; B, a edição de seis páginas; C, a implementação operacional. D–G fundamentam as distinções de métodos e a avaliação de explicações. A bibliografia completa da pesquisa permanece em paper/explicability/references.bib. O notebook reproduz somente as tabelas e gráficos de médias, sem treino ou inferência.

Fontes: Links completos em dados.json, roteiro.md e apresentacao.ipynb.

## Referências

[A] Cunha et al. — estudo anterior: https://github.com/Lucas-PG/FaceForgery-Benchmark/blob/e9beb4e212b7e3f749f65e84d27ef7f10b52de99/paper/original-sibgrapi.pdf
[B] Manuscrito principal — seis páginas: https://github.com/Lucas-PG/FaceForgery-Benchmark/blob/e9beb4e212b7e3f749f65e84d27ef7f10b52de99/paper/six-page/main.tex
[C] Guia de execução e configuração: https://github.com/Lucas-PG/FaceForgery-Benchmark/blob/e9beb4e212b7e3f749f65e84d27ef7f10b52de99/docs/explicability.md
[D] Sundararajan et al. — Integrated Gradients: https://proceedings.mlr.press/v70/sundararajan17a.html
[E] Selvaraju et al. — Grad-CAM: https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
[F] Adebayo et al. — Sanity Checks for Saliency Maps: https://arxiv.org/abs/1810.03292
[G] Hedström et al. — Quantus: https://jmlr.org/papers/v24/22-0142.html
