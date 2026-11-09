"""Conteúdo visual e notas de apresentação. Nenhum mapa de ativação é simulado."""

def compose(d, data):
    sld, t, r, ln, dot = d.slide, d.text, d.rect, d.line, d.dot
    INK, MUTED, TEAL, BLUE, WHITE, PALE = d.INK, d.MUTED, d.TEAL, d.BLUE, d.WHITE, d.PALE
    rows=data['rgb']

    s=sld('Detecção de falsificações faciais', 'MFFI · RESULTADOS E EXPLICABILIDADE',
          notes='Apresente primeiro os resultados do benchmark anterior. Em seguida, explique os três protocolos implementados para investigar as decisões dos detectores. Esta apresentação não contém heatmaps novos nem resultados inferidos como se tivessem sido medidos. Autores mantidos conforme o manuscrito fornecido; o material requer revisão da equipe.')
    t(s,'Acertar não é o\nmesmo que explicar.',80,280,950,78,bold=True)
    t(s,'Resultados do benchmark espacial–espectral\ne uma extensão para comparar explicações.',84,513,910,34,color=MUTED)
    r(s,1110,270,410,330,PALE,radius=24)
    t(s,'DUAS LIDERANÇAS',1140,301,350,19,color=TEAL,bold=True)
    t(s,'0,884',1140,351,350,69,bold=True)
    t(s,'Xception · teste limpo',1140,445,350,25,color=MUTED)
    t(s,'0,726',1140,493,180,44,color=TEAL,bold=True)
    t(s,'DINOv3 · degradado',1140,556,350,23,color=MUTED)
    t(s,'Lucas Cunha · Lucas Sotomaior · Lucas Gasperin',84,677,1380,25)
    t(s,'Beatriz Caldas · Eduardo Pianovski · Rayson Laroca',84,716,1380,25)
    t(s,data['instituicao'],84,763,1380,21,color=MUTED)

    s=sld('O que está sendo comparado',source='A · seções III–IV; B · seção III.',
          notes='O MFFI contém, no estudo anterior, 524.429 imagens de treino, 147.363 de validação e 181.947 de teste, além do teste degradado correspondente. Os resultados são médias de três sementes. Cinco famílias foram treinadas do zero; a configuração DINOv3 do estudo anterior tinha o backbone congelado. Não confundir esses modelos com o release fine-tuned usado pela extensão. ROC-AUC resume a ordenação entre classes ao variar o limiar; não é a porcentagem de acertos em um limiar específico.')
    for x,val,label,desc in [(80,'6','famílias de modelos','CNNs, transformers e DINOv3'),(580,'181.947','imagens no teste','Comparação limpa e degradada'),(1080,'3','sementes por configuração','Médias reportadas no estudo')]:
        r(s,x,280,440,310,WHITE,radius=20)
        t(s,val,x+28,317,384,68,bold=True,color=TEAL)
        t(s,label,x+28,411,384,29,bold=True)
        t(s,desc,x+28,511,384,25,color=MUTED)
    r(s,80,645,1440,110,PALE,radius=18)
    t(s,'ROC-AUC',109,671,260,34,bold=True)
    t(s,'0,50: referência aleatória  ·  1,00: separação perfeita',403,675,1070,29)

    s=sld('A degradação muda o ranking',
          notes='Todos os pontos são médias RGB do estudo anterior. Círculo azul indica teste limpo; quadrado verde indica teste degradado. O eixo começa em 0,50 e termina em 1,00, de forma explícita: é um gráfico de posições, não de barras. Xception lidera o limpo, enquanto DINOv3 congelado lidera o degradado entre as seis configurações mostradas. A figura não adiciona intervalos de confiança nem prova superioridade estatística. DINOv3 e os demais modelos diferem também no regime de pré-treinamento.')
    dot(s,102,233,16,BLUE); t(s,'Teste limpo',126,215,260,25,color=BLUE)
    dot(s,426,233,16,TEAL,True); t(s,'Teste degradado',450,215,380,25,color=TEAL)
    xmin,xmax=500,1430
    def px(v): return xmin+(v-.5)/.5*(xmax-xmin)
    for tick in [.5,.6,.7,.8,.9,1.0]:
        ln(s,px(tick),281,px(tick),740)
        t(s,d.num(tick,2),px(tick)-25,753,86,20,color=MUTED)
    ordered=sorted(rows,key=lambda a:-a['limpo'])
    for i,row in enumerate(ordered):
        y=306+i*80
        label=row['modelo']+(' (congelado)' if row['modelo']=='DINOv3' else '')
        t(s,label,80,y-19,401,27,bold=row['modelo'] in ['DINOv3','Xception'])
        ln(s,px(row['limpo']),y,px(row['degradado']),y,'#B7C4D3',4)
        dot(s,px(row['limpo']),y,17,BLUE)
        dot(s,px(row['degradado']),y,17,TEAL,True)
        t(s,d.num(row['limpo']),px(row['limpo'])-32,y-33,115,22,color=BLUE)
        t(s,d.num(row['degradado']),px(row['degradado'])-32,y+12,115,22,color=TEAL)

    s=sld('Pequena queda, sozinha, não é robustez',
          notes='A queda absoluta do ViT é menor, mas seu ponto de partida já é baixo: 0,624 no limpo e 0,608 no degradado. DINOv3 cai mais, 0,083, porém mantém AUC degradada maior, 0,726. Portanto, classificar robustez apenas pela menor queda pode favorecer um modelo pouco discriminativo. Estes são contrastes descritivos; não se isola a arquitetura do pré-treinamento.')
    for x,name,a,b in [(80,'DINOv3 congelado',.809,.726),(830,'ViT do zero',.624,.608)]:
        r(s,x,278,690,423,WHITE,radius=22)
        t(s,name,x+32,314,626,35,bold=True)
        t(s,'LIMPO',x+32,395,245,19,color=MUTED,bold=True)
        t(s,'DEGRADADO',x+361,395,290,19,color=MUTED,bold=True)
        t(s,d.num(a),x+32,435,250,68,bold=True,color=BLUE)
        t(s,d.num(b),x+361,435,280,68,bold=True,color=TEAL)
        ln(s,x+32,548,x+658,548)
        t(s,'Queda absoluta',x+32,580,340,28,color=MUTED)
        t(s,d.num(d.delta(a,b))+' AUC',x+366,574,290,36,bold=True)
    t(s,'Compare o nível de desempenho e a queda — não apenas a estabilidade.',85,740,1430,30,color=TEAL,bold=True)

    s=sld('Frequência: complemento, não substituição',
          notes='Três comparações descritas no artigo anterior: Xception RGB+magnitude melhora de 0,609 para 0,650; ResNet híbrido de 0,635 para 0,648; MobileNet RGB+magnitude de 0,594 para 0,612 no teste degradado. Os ganhos são 0,041, 0,013 e 0,018 unidades de AUC, calculados das médias arredondadas. São exemplos reportados, não todos os modos nem uma seleção feita em validação nesta apresentação. Não chamar AUC de porcentagem de acerto nem os ganhos de prova causal.')
    t(s,'Três contrastes reportados no teste degradado',80,208,1390,29,color=MUTED)
    dot(s,635,277,14,BLUE); t(s,'RGB',660,258,145,22,color=BLUE)
    dot(s,848,277,14,TEAL,True); t(s,'Híbrido reportado',873,258,360,22,color=TEAL)
    t(s,'GANHO',1370,258,145,19,color=MUTED,bold=True)
    def hx(v): return 625+(v-.5)/.25*670
    for tick in [.5,.55,.6,.65,.7,.75]:
        ln(s,hx(tick),315,hx(tick),701)
        t(s,d.num(tick,2),hx(tick)-25,718,85,19,color=MUTED)
    for i,row in enumerate(data['hibridos']):
        y=351+i*140
        t(s,row['modelo'],80,y-35,468,32,bold=True)
        t(s,row['representacao'],80,y+13,470,23,color=MUTED)
        a,b=row['rgb_degradado'],row['hibrido_degradado']
        ln(s,hx(a),y,hx(b),y,'#B7C4D3',4)
        dot(s,hx(a),y,17,BLUE); dot(s,hx(b),y,17,TEAL,True)
        t(s,d.num(a),hx(a)-28,y+14,100,22,color=BLUE)
        t(s,d.num(b),hx(b)-28,y-38,100,22,color=TEAL)
        t(s,'+'+d.num(d.delta(b,a)),1364,y-20,154,37,color=TEAL,bold=True)
    t(s,'Ganhos descritivos em AUC; não demonstram qual artefato foi aprendido.',80,768,1430,23,color=MUTED)

    s=sld('O que os resultados permitem concluir',
          notes='Separar observação de mecanismo: o ranking muda e existem ganhos em entradas híbridas selecionadas no artigo. Esses resultados motivam investigar evidência local ou espectral, mas não demonstram que uma rede identificou um tipo específico de artefato. Tampouco isolam efeitos de arquitetura, pré-treinamento, identidade ou fonte. As análises de explicabilidade entram para formular e examinar essas hipóteses.')
    d.paragraph_card(s,80,285,690,368,'OBSERVAÇÃO','O ranking muda.','Há queda sob degradação e ganhos em algumas entradas híbridas.')
    d.paragraph_card(s,830,285,690,368,'INTERPRETAÇÃO','A causa não está isolada.','Arquitetura, pré-treinamento e pistas visuais ainda precisam ser separados.',fill=PALE)
    t(s,'Um mapa plausível não é, por si só, uma explicação fiel.',85,726,1430,35,bold=True,color=TEAL)

    s=sld('Três perguntas para explicar as decisões','EXTENSÃO IMPLEMENTADA · EXPERIMENTOS PENDENTES','B · seção IV; C · guia de execução.',
          notes='A partir daqui, apresenta-se o protocolo implementado, não resultados recém-observados. A primeira tarefa controla as imagens; a segunda controla a comparação RGB–frequência; a terceira define a população de falhas compartilhadas. Todas preservam rastreabilidade, limiares congelados e limitações de interpretação.')
    cards=[('01','Mesmas imagens','Como as arquiteturas explicam os mesmos acertos e erros?'),('02','RGB e frequência','As explicações mudam quando muda o domínio de entrada?'),('03','Falhas compartilhadas','Que condições aparecem quando todos os modelos do grupo erram?')]
    for i,(label,headline,body) in enumerate(cards):
        x=80+i*500
        r(s,x,298,440,383,WHITE,radius=22)
        t(s,label,x+30,329,380,43,color=TEAL,bold=True)
        t(s,headline,x+30,410,380,33,bold=True)
        t(s,body,x+30,525,380,29,color=MUTED)
    t(s,'Comparações controladas; nenhum resultado de heatmap é antecipado.',85,742,1430,28,color=TEAL)

    s=sld('Tarefa 1 · As mesmas 64 imagens','PROTOCOLO · NÃO É UM RESULTADO EXPERIMENTAL','B · tarefa 1; C · configuração.',
          notes='A classe positiva é falsa. O modelo de referência é ResNet/RGB/seed 42; a semente de amostragem é 42. VP: falsa prevista falsa; FN: falsa prevista real; VN: real prevista real; FP: real prevista falsa. São 16 casos sem reposição em cada estrato. As mesmas identidades e ordem são reutilizadas; outros modelos mantêm seus próprios desfechos. Essa amostra super-representa erros, portanto não estima acurácia populacional nem prevalência de falhas.')
    for x,y,label,definition in [(80,290,'VP','Falsa prevista falsa'),(450,290,'FN','Falsa prevista real'),(80,510,'VN','Real prevista real'),(450,510,'FP','Real prevista falsa')]:
        r(s,x,y,340,194,WHITE,radius=18)
        t(s,label,x+27,y+23,260,24,color=TEAL,bold=True)
        t(s,'16',x+27,y+61,260,65,bold=True)
        t(s,definition,x+27,y+149,290,22,color=MUTED)
    t(s,'64',933,285,500,126,color=TEAL,bold=True)
    t(s,'Mesmas identidades.\nMesma ordem.\nTodos os modelos.',936,469,550,39,bold=True)
    t(s,'O equilíbrio 16×4 vale para o modelo de referência, não para todos simultaneamente.',84,752,1430,27,color=MUTED)

    s=sld('Tarefa 2 · Comparar no domínio correto','PROTOCOLO · NÃO É UM RESULTADO EXPERIMENTAL','B · tarefa 2; C · seleção e renderização.',
          notes='A comparação exige checkpoints separadamente treinados, da mesma família, regime e semente, em RGB e frequência. A seleção usa apenas validação e um piso de competência, não o ranking do teste. A implementação compara as mesmas 64 imagens, com atribuições em pixels sobre RGB e em bins sobre a entrada de frequência. Este slide é um esquema textual, sem heatmaps simulados. Um coeficiente global de Fourier não corresponde a uma posição facial; não calcular sobreposição espacial RGB–espectro como se fossem as mesmas coordenadas.')
    d.paragraph_card(s,80,302,690,365,'ESPACIAL','Onde na imagem?','Pixels da imagem RGB.\nAtribuição sobre a entrada espacial.')
    d.paragraph_card(s,830,302,690,365,'FREQUÊNCIA','Quais frequências?','Coeficientes do espectro.\nAtribuição sobre a entrada espectral.',fill=PALE)
    t(s,'Um agrupamento no espectro não é uma localização no rosto.',84,735,1430,32,color=TEAL,bold=True)

    s=sld('Tarefa 3 · Falhas do conjunto declarado','PROTOCOLO · NÃO É UM RESULTADO EXPERIMENTAL','B · tarefa 3; C · controles e revisão humana.',
          notes='A interseção exige todos os modelos no mesmo conjunto de imagens, com IDs únicos, rótulos consistentes e predições completas. A população primária usa seis checkpoints RGB seed 42; a análise de sensibilidade usa os 18 checkpoints de três sementes. São mantidas todas as falhas compartilhadas, sem teto oculto. Os controles são pareados por rótulo e não pertencem à interseção; podem falhar em parte dos modelos. Iluminação e oclusão exigem anotação e comparação; rótulo pareado não controla identidade, fonte ou compressão. Não inferir demografia automaticamente.')
    for i,(label,head,body) in enumerate([('01 · ALINHAMENTO','Validar','Mesmas imagens, rótulos e predições completas.'),('02 · INTERSEÇÃO','Encontrar','Imagens erradas por todos os checkpoints declarados.'),('03 · CONTROLE','Comparar','Revisão humana cega e controles pareados por rótulo.')]):
        x=80+i*500
        d.paragraph_card(s,x,301,440,381,label,head,body)
    t(s,'Falha compartilhada não significa vulnerabilidade universal.',84,740,1430,33,color=TEAL,bold=True)

    s=sld('12 métodos, perguntas diferentes','IMPLEMENTAÇÃO · SUPORTE DEPENDE DA ARQUITETURA','B · tabela de métodos; D–G · base metodológica.',
          notes='Métodos agrupados pela forma predominante de análise: seis por gradientes/localização, cinco por perturbação/surrogates e um de atenção. Kernel SHAP e LIME produzem contribuições de grupos sob perturbação; Grad-CAM localiza em uma camada; IG é relativo a baseline. Attention Rollout é agnóstico à classe e não é equivalente à explicação por gradiente da margem falso-real. Nem todos os métodos são aplicáveis a todos os modelos. O número de métodos não corresponde a confirmações independentes.')
    groups=[('GRADIENTES / LOCALIZAÇÃO',data['metodos']['Gradientes e localização']),('PERTURBAÇÃO / MODELOS LOCAIS',data['metodos']['Perturbação e aproximação local']),('ATENÇÃO',data['metodos']['Atenção'])]
    for i,(label,methods) in enumerate(groups):
        x=80+i*500
        r(s,x,281,440,418,WHITE,radius=20)
        t(s,label,x+26,311,388,18,color=TEAL,bold=True)
        t(s,'\n'.join(methods),x+26,369,388,29,leading=1.42)
        if i==2: t(s,'Agnóstico à classe.\nNão equivale a uma\natribuição da decisão.',x+26,455,388,28,color=MUTED)
    t(s,'Alvo fixo nas atribuições de classe: logit falso menos logit real.',84,741,1430,30,color=TEAL,bold=True)

    s=sld('O que está pronto e o que falta medir','ESTADO DA ENTREGA','B · seção VI; C · execução e cobertura.',
          notes='A parte implementada inclui amostragem e seleção reproduzíveis, atribuições, exportação, hashes, verificação das predições, retomada por partições e controles numéricos. As experiências de imagem ainda precisam ser executadas para produzir as galerias e a análise de falhas. Nesta apresentação não se treina modelo, não se usa GPU e não se produz nova AUC. Testes sintéticos de software e compilação não são evidência de fidelidade das explicações em dados reais.')
    r(s,80,281,690,420,PALE,radius=20); r(s,830,281,690,420,d.SAND,radius=20)
    t(s,'IMPLEMENTADO',111,316,620,21,color=TEAL,bold=True)
    t(s,'Amostras e seleção reproduzíveis\nMétodos e exportação de mapas\nRastreabilidade e retomada\nControles e cobertura',111,388,625,31,leading=1.64)
    t(s,'AINDA A MEDIR',861,316,620,21,color='#936039',bold=True)
    t(s,'Galerias de mapas reais\nComparações RGB–frequência\nAnálise dos casos compartilhados\nAssociações com condições visuais',861,388,625,31,leading=1.64)
    t(s,'A literatura e o código orientam hipóteses; os dados decidem os resultados.',84,742,1430,28,color=MUTED)

    s=sld('Três mensagens para levar','SÍNTESE','A–C · resultados anteriores e extensão.',
          notes='Encerrar com três mensagens: avaliar no limpo não substitui avaliar sob degradação; sinais espectrais podem complementar RGB mas não revelam automaticamente o mecanismo; explicar exige controlar a amostra, o alvo, as coordenadas e os métodos. As tarefas implementadas permitem investigar essas questões sem antecipar achados. Os materiais completos e fontes estão na pasta presentation e no notebook.')
    for y,n,headline,body in [(281,'01','Robustez muda o ranking.','A melhor média no teste limpo não garante liderança após degradação.'),(447,'02','Frequência pode complementar RGB.','Ganhos reportados motivam investigar as pistas realmente utilizadas.'),(613,'03','Explicabilidade exige controle.','Mesmas imagens, alvos fixos, domínios corretos e falhas bem definidas.')]:
        t(s,n,82,y,120,53,color=TEAL,bold=True)
        t(s,headline,249,y+2,1260,41,bold=True)
        t(s,body,249,y+71,1260,27,color=MUTED)

    s=sld('Apoio · Tabela completa dos resultados RGB','RESULTADOS REPORTADOS · SEM NOVOS EXPERIMENTOS',
          notes='Tabela de consulta com todas as seis médias RGB e suas quedas. O valor na última coluna é limpo menos degradado, com Decimal para preservar a subtração das médias arredondadas. Não misturar esses números com o release atual fine-tuned. Os desvios-padrão do estudo original não são reconstruídos a partir de médias. Fontes A e B indicam a origem; dados.json permite editar e verificar o material.')
    widths=[(99,'Configuração anterior'),(730,'Limpo'),(985,'Degradado'),(1260,'Queda em AUC')]
    for x,label in widths: t(s,label,x,272,360 if x==99 else 246,23,color=MUTED,bold=True)
    ln(s,80,319,1520,319)
    for i,row in enumerate(rows):
        y=351+i*64
        if i%2==0: r(s,80,y-8,1440,58,WHITE,radius=8)
        name=row['modelo']+(' · congelado' if row['modelo']=='DINOv3' else ' · do zero')
        t(s,name,99,y,600,28,bold=row['modelo']=='DINOv3')
        t(s,d.num(row['limpo']),730,y,230,29,color=BLUE)
        t(s,d.num(row['degradado']),985,y,230,29,color=TEAL,bold=True)
        t(s,d.num(d.delta(row['limpo'],row['degradado'])),1260,y,245,29)
    t(s,'Diferenças das médias arredondadas; não representam significância estatística.',83,751,1430,26,color=MUTED)

    s=sld('Fontes e material de apoio','REFERÊNCIAS','Links completos em dados.json, roteiro.md e apresentacao.ipynb.',
          notes='Os links de documentos do repositório estão fixados no commit usado para construir a apresentação. A referência A contém o estudo original; B, a edição de seis páginas; C, a implementação operacional. D–G fundamentam as distinções de métodos e a avaliação de explicações. A bibliografia completa da pesquisa permanece em paper/explicability/references.bib. O notebook reproduz somente as tabelas e gráficos de médias, sem treino ou inferência.')
    entries=[('A','Estudo anterior','Cunha et al. · Tabela III e seção IV'),('B','Manuscrito de seis páginas','Resultados anteriores e protocolos'),('C','Guia e configuração','Execução, artefatos e limites'),('D','Integrated Gradients','Sundararajan et al. · ICML 2017'),('E','Grad-CAM','Selvaraju et al. · ICCV 2017'),('F','Sanity Checks for Saliency Maps','Adebayo et al. · NeurIPS 2018'),('G','Quantus','Hedström et al. · JMLR 2023')]
    for i,(code,head,desc) in enumerate(entries):
        col=0 if i<4 else 1; row=i if i<4 else i-4
        x=80+col*750; y=281+row*112
        t(s,code,x,y,65,32,color=TEAL,bold=True)
        t(s,head,x+70,y,605,29,bold=True)
        t(s,desc,x+70,y+48,605,24,color=MUTED)
