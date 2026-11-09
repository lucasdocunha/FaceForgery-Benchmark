# Checagem: ambiente

- ❌ `conda activate` sem `source /opt/conda/etc/profile.d/conda.sh` antes (em job não-interativo o `conda` não existe).
- ❌ env referenciado por caminho em `/home/$USER/...` — é disco local de um nó; o job que cair no outro nó não acha. Envs só em `$HOME/.conda/envs` (`/users/home/$USER`), ativados por nome. Entradas em `/home/$USER` que aparecem no `conda env list` são restos do `environments.txt`, não envs válidos.
- ❌ `module load` sem shell de login (shebang sem `-l` e sem `bash -lc`).
- ❌ `sudo`, `docker`, `apt install` em qualquer lugar do job ou do setup (sem sudo; exceção: `sudo ngc-pull`). Container → Apptainer.
- ❌ `pip install`/`conda install` dentro do job a cada execução — o env é preparado antes, no Shell Access.
- ⚠️ imports do código que não se sabe se estão no env: liste-os e ponha em "Comandos pra rodar": `conda list -n <env> | grep -iE '<pacote1>|<pacote2>'`.
- ⚠️ PyTorch instalado sem build CUDA (`cpu`) ou com CUDA antigo — o padrão é `cu128`.
- ⚠️ caminho absoluto de outra máquina no código ou config (`/home/<outro>`, `C:\`, `/mnt/c`, caminho da workstation) — tem que existir no cluster ou vir de argumento/variável.
