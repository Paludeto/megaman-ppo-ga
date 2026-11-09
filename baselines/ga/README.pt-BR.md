# Algoritmos genéticos

*Leia em outro idioma: [English](README.md) · **Português***

Dois algoritmos genéticos que jogam a mesma tarefa do agente PPO:

| Método | Genoma | Percepção | Referência |
|--------|--------|-----------|------------|
| **GA de malha aberta** (GA-OL) | sequência de ações `MultiDiscrete([2,5,2])` em blocos de `k` passos do agente | nenhuma: repete as mesmas ações, aconteça o que acontecer na tela | a ideia do "Brute" de Machado et al. 2018 |
| **Deep GA** | pesos de uma `NatureCNN` + cabeça linear com 2+5+2 logits (1.688.745 parâmetros), ação = argmax por cabeça | a mesma pilha de quadros uint8 4×84×84 do PPO | Such et al. 2017 |

As tarefas são montadas com `src.env.make_env` (frameskip 4, 7200 quadros do emulador).

- **Fitness:** retorno não descontado do episódio na recompensa de treino do PPO: `+0,05` por acerto
  no chefe, `-0,10` por evento de dano e `+0,5` pelo abate. O desconto e a normalização de recompensa
  do PPO pertencem ao algoritmo e não são aplicados.
- **Empates:** desfeitos por vitória e depois por menos quadros até o abate.
- **Avaliação:** `0,05 * (acertos no chefe - eventos de dano)`, pelo mesmo `src.eval.evaluate()` usado
  no PPO. A vitória exige o chefe morto com o Mega Man vivo.

Execute tudo a partir da raiz do repositório, com um `rom.nes` legalmente obtido. Cada processo
trabalhador tem um emulador e uma thread de torch/OpenCV; são permitidos no máximo 12 processos.

## Uso

O `fair.py` conta os passos do agente de todos os rollouts em `--max-env-steps`.

```bash
# treino determinístico: para na primeira validação sem dano
python -m baselines.ga.fair --variant deep-ga --boss cutman --max-env-steps 3000000 --seed 0 --out-dir out/dga

# treino com sticky actions: K episódios por indivíduo, vai até o orçamento
python -m baselines.ga.fair --variant ga-ol --boss cutman --max-env-steps 3000000 --seed 0 --out-dir out/ol \
    --train-sticky 0.25 --episodes-per-eval 1 --no-early-stop

# avaliação de um indivíduo salvo (best, final ou snapshot_<passo>)
python -m baselines.ga.evaluate --run out/dga --which final --episodes 30 --sticky 0.25 --seed 123 --json eval.json
```

| Regime | Rollouts | Elites | Parada |
|--------|----------|--------|--------|
| Determinístico | um episódio exato por indivíduo | mantêm suas métricas | primeira validação sem dano, a cada `--validate-every` passos (100 mil), ou o orçamento |
| Sticky (`--train-sticky P`) | `--episodes-per-eval K` episódios com sementes novas; o fitness é a média | reavaliados a cada geração | o orçamento |

O diretório de `--out-dir` recebe `config.json`, `history.json`, `validations.json`, `best.json`,
`final.json`, `snapshot_<passo>.json` (com `--snapshot-at`) e `summary.json`.

| | GA-OL | Deep GA |
|--|-------|---------|
| População (`--pop`) | 240 | 96 |
| Seleção | truncamento em `--trunc` 0,2, `--elite` 6 | truncamento em `--parents` 10, `--elite` 1 |
| Variação | crossover de dois pontos (`--p-cross` 0,5), reamostragem por gene (`--p-mut` 0,02) | mutação gaussiana (`--sigma` 0,005), sem crossover |
| Genoma | blocos de `--k` 4 passos do agente | cadeia de sementes, `theta_g = theta_(g-1) + sigma * eps(seed_g)` |

Esses são os valores padrão. Os hiperparâmetros selecionados para cada chefe estão em
`experiments/fair/selection/`, e o protocolo está em [docs/EXPERIMENTS.pt-BR.md](../../docs/EXPERIMENTS.pt-BR.md).
