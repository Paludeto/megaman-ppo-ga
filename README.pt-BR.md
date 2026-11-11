# Chefes de Mega Man (NES): PPO, Deep GA e GA de malha aberta

*Leia em outro idioma: [English](README.md) · **Português***

Código de uma comparação de três métodos nas 6 lutas contra Robot Masters de *Mega Man* (NES): Gutsman,
Cutman, Bombman, Elecman, Iceman e Fireman.

| Método | O que é aprendido | Observação | Código |
|--------|-------------------|------------|--------|
| **PPO** (stable-baselines3, `CnnPolicy`) | pesos da rede, por gradiente | pilha de quadros 84×84×4 em tons de cinza | `src/` |
| **Deep GA** (Such et al. 2017) | pesos da mesma CNN, por mutação e seleção | a mesma pilha de quadros | `baselines/ga/` |
| **GA de malha aberta** (a ideia do "Brute" de Machado et al. 2018) | uma sequência fixa de ações | nenhuma: nunca vê a tela | `baselines/ga/` |

`experiments/fair/` executa os três sob um único protocolo: mesmo orçamento de passos, recompensa,
esforço de ajuste e avaliador. **Para reproduzir a campanha, veja
[docs/EXPERIMENTS.pt-BR.md](docs/EXPERIMENTS.pt-BR.md).**

- **Ações:** `MultiDiscrete([2, 5, 2])` = `[B, D-pad, A]`; os valores do D-pad são nada, UP, DOWN,
  LEFT, RIGHT.
- **Recompensa de avaliação:** `r = d · (acertos − dano)`, com `d = 0,05`. Acertos e dano são
  *eventos* (um passo em que o chefe ou o Mega Man perdeu HP), não magnitudes de HP.
- **Recompensa de treino:** a mesma, com penalidade de dano 2× e `+0,5` pelo abate. Os GAs usam o
  retorno não descontado dela como fitness.
- **Vitória:** `boss_health` (endereço de RAM 1729) chega a 0 com o Mega Man vivo.

## Preparação

```bash
conda env create -f environment.yml && conda activate megaman-ppo
cp /caminho/para/sua/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes
tail -c +17 custom_integrations/MegaMan-v1-Nes/rom.nes | sha1sum
# deve ser: 6047e52929dfe8ed4708d325766ccb8d3d583c7d
pytest tests -q
```

O `environment.yml` instala as versões exatas de `requirements.lock.txt` (Python 3.12.13, Linux, wheels
CUDA 13). Uma GPU é fortemente recomendada para o PPO; os GAs rodam em CPU, com um emulador por
processo trabalhador (no máximo 12).

> A ROM **não** é distribuída aqui. Você precisa de uma cópia legalmente obtida de *Mega Man (NES)*. O
> `rom.sha` guarda o SHA-1 **sem** o cabeçalho iNES de 16 bytes, a convenção do stable-retro. O
> `make_env` o verifica e levanta `RuntimeError` se não bater. Os testes que precisam do emulador são
> ignorados sem `rom.nes`.

## Início rápido

```bash
python src/train.py --tag bombman --state Bombman-boss --timesteps 3000000 --lr-horizon 40000000
python src/eval.py --model models/bombman_best/final_model.zip --state Bombman-boss \
  --episodes 30 --sticky 0.25 --seed 123

python -m baselines.ga.fair --variant deep-ga --boss bombman \
  --max-env-steps 3000000 --seed 0 --out-dir out/deep-ga
python -m baselines.ga.evaluate --run out/deep-ga --which final \
  --episodes 30 --sticky 0.25 --seed 123 --json out/deep-ga/eval.json
```

Use `--variant ga-ol` para o GA de malha aberta. Execute tudo a partir da raiz do repositório.

## Estrutura

```
src/
  wrappers.py        frameskip, recompensa (BossWrapper), pré-processamento 84×84
  env.py             make_env / make_venv: integração, checagem da ROM, vetorização
  train.py           treino do PPO
  eval.py            avaliação: intervalos de Wilson, sticky actions, no-op starts
baselines/ga/
  fair.py            laço de busca dos GAs (as duas variantes)
  evaluate.py        avalia um indivíduo salvo por meio do src/eval.py
  deep_ga.py, ga_ol.py   operadores genéticos
  policy.py, common.py   indivíduos como políticas; tarefas, processos, execução de episódios
experiments/fair/
  config.py, jobs.py, scheduler.py, status.py, stop.sh   a campanha e seu grafo de jobs
  tune_select.py, selection/, use_selection.py           seleção de hiperparâmetros
  aggregate.py, stats.py, ppo_summary.py                 estatística e tabelas
  random_baseline.py                                     política aleatória exploratória
tests/               suíte pytest
custom_integrations/MegaMan-v1-Nes/   integração do stable-retro (sem rom.nes)
docs/                guia de arquitetura e guia da campanha
```

Gerados em execução e ignorados pelo git: `logs/`, `checkpoints/`, `models/`, `out/`, `campaign/`.

## Treinar o PPO

A receita é o conjunto de constantes no topo de `src/train.py`. A cada ~100 mil passos a política joga
um episódio determinístico de validação na tarefa de avaliação, e o `best_model.zip` guarda a melhor
recompensa de validação. O treino termina na primeira validação vencida **sem dano** ou no teto
`--timesteps`; com `--no-early-stop`, sempre vai até o teto.

| Flag | Significado |
|------|-------------|
| `--tag`, `--state` | nome da execução; save state, como `Bombman-boss` (obrigatório) |
| `--timesteps`, `--n-envs`, `--seed` | teto (40M), ambientes paralelos (16), semente (666) |
| `--lr`, `--ent-coef`, `--lr-horizon` | LR inicial (2,5e-4), coeficiente de entropia (5e-3), passo em que a LR chega a 0 |
| `--train-sticky P`, `--no-early-stop` | sticky actions no treino (a validação continua determinística); sem parada antecipada |
| `--episode-frames` | duração do episódio em quadros do emulador (7200) |
| `--deterministic-torch` | kernels determinísticos de torch e CUDA (mais lento) |

Cada execução grava `logs/<tag>/` (TensorBoard e `run_meta.json`), `checkpoints/<tag>/` e
`models/<tag>_best/{best_model,final_model}.zip`. O `run_meta.json` registra versões das bibliotecas,
commit, hardware, argumentos e constantes.

## Avaliar

O `src/eval.py` reporta taxas de vitória e de vitória sem dano com intervalos de Wilson de 95%, HP
perdido, eventos de dano e quadros até o abate. Opções: `--deterministic` (padrão) ou `--stochastic`,
`--sticky P` (cada quadro do emulador repete a ação anterior com probabilidade `P`), `--noop-max N`,
`--seed`, `--device`, `--episode-frames`, `--json CAMINHO`. Um único episódio determinístico de
validação nada diz sobre robustez; meça isso aqui.

## Documentação

- [docs/EXPERIMENTS.pt-BR.md](docs/EXPERIMENTS.pt-BR.md): protocolo e reprodução da campanha.
- [RESULTS.md](RESULTS.md): resultados da campanha original, os mesmos do artigo (em inglês).
- [docs/ARCHITECTURE.pt-BR.md](docs/ARCHITECTURE.pt-BR.md): como o ambiente, o PPO e os GAs são construídos.
- [baselines/ga/README.pt-BR.md](baselines/ga/README.pt-BR.md): os algoritmos genéticos.

## Licença

Veja [LICENSE](LICENSE). A ROM de *Mega Man* não está incluída e não é coberta por esta licença.
