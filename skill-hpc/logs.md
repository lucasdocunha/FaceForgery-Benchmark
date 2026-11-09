# Checagem: logs

Todo job precisa ter logs pra mostrar. Tudo vai para stdout/stderr (o SLURM grava em `logs/%x_%j.out` / `.err`); o código **não** abre arquivo de log próprio no NFS.

## Conteúdo obrigatório do `.out`

1. **Cabeçalho**: job id, nome, nó, partição, GPU, env conda, commit do git, argumentos, hora de início. (O [template.sh](template.sh) já imprime o que é do ambiente; o código imprime a sua configuração efetiva.)
2. **Progresso**: a cada N itens/steps — feitos/total, taxa (itens/s) e ETA. Com `print(..., flush=True)` ou `logging` para stdout; barras `tqdm` em arquivo viram lixo — preferir `tqdm(..., mininterval=60)` ou log por linha.
3. **Rodapé**: o que foi gravado e onde (caminhos, contagens), duração, código de saída.
4. **Métricas de saída**, quando o job produz alguma (acurácia, loss final, F1, itens ok/erro): impressas no fim, legíveis.

## Bloqueia (❌)

- `--output` apontando para `logs/` que não existe.
- Código gravando log em arquivo no NFS/`/home` em vez de stdout.

## Aviso (⚠️)

- Falta cabeçalho, progresso, rodapé ou métricas (diga qual e onde inserir, `arquivo:linha`).
- Saída bufferizada (Python sem `-u`/`flush=True`/`PYTHONUNBUFFERED=1`): o `.out` fica vazio até o fim e não dá pra acompanhar.
- Exceção engolida (`except: pass`) — o rodapé mostraria código 0 com falha.
- `.out`/`.err` na raiz do projeto (acumula centenas de arquivos) em vez de `logs/`.
