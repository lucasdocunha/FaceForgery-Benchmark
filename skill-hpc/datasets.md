# Checagem: datasets

`/datasets` é NFS compartilhado por todos os usuários do cluster; datasets ficam **sempre** lá.

## Layout de um dataset

```
/datasets/<nome_descritivo>/
├── README.md        # origem, licença, dono ($USER), data, formato, contagem de itens
├── lote_001/        # ≤ 1.000 arquivos por pasta
├── lote_002/
└── ...
```

## Bloqueia (❌)

- Job escreve em `/datasets` (o job só lê; criar/atualizar dataset é passo separado e consciente).
- Dataset lido de caminho fora do cluster (workstation, `/mnt/c`, URL que o código baixa a cada execução quando o dado deveria estar em `/datasets`).

## Aviso (⚠️)

- Dataset novo sem `README.md` ou com milhares de arquivos soltos numa pasta.
- Saída intermediária/derivada do job sendo gravada como "dataset" — vai para `saidas/` do projeto; só o dataset **final** entra em `/datasets`.
- Nome genérico (`dados`, `teste`, `novo`) — o nome identifica o conteúdo.
- Leitura de milhares de arquivos pequenos por item (I/O pesado no NFS): sugerir empacotar (parquet, tar/webdataset) ou copiar para `$TMPDIR` no início do job.
