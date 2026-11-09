# Campanha de comparação: PPO, Deep GA e GA de malha aberta

*Leia em outro idioma: [English](EXPERIMENTS.md) · **Português***

`experiments/fair/` treina e avalia os três métodos sob um único protocolo. Todas as constantes estão
em `experiments/fair/config.py`.

## Protocolo

| Item | Valor |
|------|-------|
| Chefes | Gutsman, Cutman, Bombman, Elecman, Iceman, Fireman |
| Orçamento | 3.000.000 de passos do agente por execução; todos os passos de rollout contam |
| Sementes | PPO 666, 1, 2, 3, 4; GAs 0, 1, 2, 3, 4 |
| Episódio | no máximo 7200 quadros do emulador (1800 passos do agente) |
| Execuções finais | 3 métodos × 6 chefes × 2 regimes × 5 sementes = 180 |

**Regimes de treino**

- `sticky` (primário): sticky actions com `p = 0,25` por quadro do emulador em todos os rollouts de
  treino, sem parada antecipada.
- `det`: treino determinístico que para na primeira validação sem dano, verificada a cada 100 mil passos.

Nos dois regimes o modelo avaliado é o final: a política na parada ou no fim do orçamento. Nos GAs é o
melhor indivíduo da última geração, sem reavaliação. Os episódios de validação do PPO não contam para o
orçamento, pois o `src/train.py` os joga em um ambiente separado. Os GAs leem a validação dos próprios
rollouts.

**Ajuste de hiperparâmetros.** Cada método tem uma grade de 4 configurações (`DET_GRID`), executada nos
6 chefes com a semente 1000, no regime `det` e com o orçamento da campanha. As execuções são ordenadas
por (chegou à parada sem dano, menos passos até ela, recompensa final de validação). As configurações
são comparadas pelo posto médio em um fold de chefes, A = {Gutsman, Cutman, Elecman} e B = {Bombman,
Iceman, Fireman}, e cada chefe treina com a escolha do **outro** fold.

No regime `sticky` os GAs também ajustam `K ∈ {1, 4}`, os episódios por indivíduo, que não têm
equivalente no PPO. São usados os mesmos folds e a mesma semente; o critério é a recompensa média de
treino do indivíduo final em 30 episódios com sticky actions.

**Avaliação.** `src.eval.evaluate()` é usado em todos os métodos, com a semente 123 e as mesmas
sequências por episódio.

| Condição | `p` sticky | No-op máx. | Episódios | Política |
|----------|:----------:|:----------:|:---------:|----------|
| `det1` | 0 | 0 | 1 | gulosa |
| `sticky` (primária) | 0,25 | 0 | 30 | gulosa |
| `stickynoop` | 0,25 | 30 | 30 | gulosa |
| `noop` | 0 | 30 | 30 | gulosa |
| `stickystoch` (só PPO) | 0,25 | 0 | 30 | amostrada |

As curvas de aprendizado avaliam a política em `sticky` a cada 500 mil passos.

**Estatística** (`experiments/fair/stats.py`). A unidade de análise é a semente de treino, pontuada pela
taxa de vitória nos 30 episódios de uma condição.

- Por chefe: média com intervalo bootstrap percentílico de 95% (10.000 reamostragens) e teste exato de
  permutação bilateral nas 252 divisões, com correção de Holm nos 6 chefes por par de métodos e regime.
  Com 5 sementes por método, o menor `p` ajustado por Holm possível é 0,048.
- Em todos os chefes (teste primário): teste de permutação estratificado da diferença média por chefe
  (100.000 permutações), com correção de Holm nas duas comparações (PPO contra cada GA). O limite de
  resolução é `p = 2e-5` após a correção.
- Média interquartil e probabilidade de melhoria, com intervalos por bootstrap estratificado.
- Exploratórios e sem correção: PPO contra Deep GA em 0,5M e 1M passos no regime `sticky`
  (`early_budget_tests` em `results.json`) e o piso da política aleatória (`random_baseline.py`).

## Como reproduzir a campanha

```bash
conda env create -f environment.yml && conda activate megaman-ppo     # versões exatas
cp /caminho/para/sua/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes

python -m experiments.fair.use_selection    # usa os hiperparâmetros da campanha original (pula 96 execuções de ajuste)
mkdir -p campaign/fair
setsid nohup python -m experiments.fair.scheduler >> campaign/fair/scheduler.out 2>&1 &

python -m experiments.fair.status           # jobs por estado, o que está rodando, falhas
experiments/fair/stop.sh                    # para; os jobs em andamento são refeitos no próximo início
python -m experiments.fair.random_baseline  # depois da campanha: piso exploratório da política aleatória
```

Execute a partir da raiz do repositório, no ambiente conda: os jobs usam o interpretador que inicia o
escalonador. O último job é o `experiments.fair.aggregate`, que grava as tabelas e a estatística;
`python -m experiments.fair.aggregate --partial` as calcula com as sementes já concluídas.

`experiments/fair/selection/` guarda as configurações escolhidas na campanha original. Sem o
`use_selection`, o ajuste também é refeito; a campanha passa a ter 1362 jobs (96 execuções de ajuste,
5 seleções, 180 execuções finais, 180 avaliações, 900 pontos de curva e a agregação), e o ajuste do PPO
pode selecionar outras configurações (veja abaixo).

O escalonador é retomável. Um job está concluído quando existe `campaign/fair/markers/<id>.done`; uma
falha grava `<id>.failed` e bloqueia os dependentes até esse arquivo ser apagado. `CPU_SLOTS`,
`GPU_SLOTS`, `MAX_GA` e `SLOTS` em `config.py` foram dimensionados para a máquina abaixo, então ajuste
para a sua. Eles alteram o tempo de execução, não os resultados. Nessa máquina, o log do escalonador da
campanha abrange cerca de 33 horas.

Para testar todo o pipeline em miniatura (2 chefes, 2 sementes, orçamentos mínimos, saída em
`campaign/fair_smoke`), nunca para gerar resultados:

```bash
mkdir -p campaign/fair_smoke
FAIR_SMOKE=1 python -m experiments.fair.scheduler
```

### O que foi verificado

Com este código e o ambiente travado, na máquina da campanha original (AMD Ryzen 9 7950X, NVIDIA RTX
4070 Ti SUPER, driver 580.178.04, Linux), uma amostra da campanha foi executada de novo e comparada com
as saídas originais:

| Verificação | Resultado |
|-------------|-----------|
| 3 execuções de GA até 3M passos (Deep GA `det` e `sticky`, GA de malha aberta `sticky`) | histórico, snapshots e indivíduos finais idênticos |
| 1 execução de PPO até 500 mil passos (`sticky`) | pesos da política idênticos ao checkpoint original |
| 5 avaliações de modelos salvos (PPO guloso e amostrado, os dois GAs) | idênticas episódio a episódio |
| Estatística recalculada a partir das avaliações originais | idêntica |
| Piso da política aleatória | idêntico |

A campanha completa não foi refeita com este código exato. Só se espera resultado idêntico bit a bit nas
mesmas condições: os GAs rodam em CPU e dependem do emulador, das sementes e das versões das bibliotecas,
enquanto o PPO depende também da GPU, do driver e dos kernels CUDA, pois a campanha não passa
`--deterministic-torch`. Em outro hardware, espere que o PPO divirja das execuções originais e compare
estatisticamente, entre sementes. Com `--deterministic-torch`, duas execuções curtas de PPO nessa máquina
tiveram pesos idênticos, mas esse não é o modo usado na campanha.

## Saídas

Tudo abaixo é gerado e ignorado pelo git.

| Caminho | Conteúdo |
|---------|----------|
| `campaign/fair/tuning/` | execuções de ajuste e `chosen_<regime>_<método>.json` |
| `campaign/fair/runs/<regime>/<método>/<chefe>/s<semente>/` | execuções de GA |
| `models/fair_<regime>_<chefe>_s<semente>*_best/` | modelo final do PPO e snapshots da curva de aprendizado |
| `campaign/fair/eval/` | um JSON por modelo e condição |
| `campaign/fair/results.json`, `agarwal.json` | escores por semente, testes, curvas de aprendizado, IQM e probabilidade de melhoria |
| `campaign/fair/RESULTS.md` | tabelas, com rótulos em português |
| `campaign/fair/random_baseline.json` | vitórias da política aleatória por chefe, semente e condição |
| `campaign/fair/logs/`, `markers/`, `scheduler.log` | logs por job e estado |

## Limites de interpretação

- O orçamento conta passos do agente, não quadros do emulador nem custo computacional.
- Cada chefe começa de um único save state fixo, e cada condição tem 30 episódios.
- Por padrão, o PPO treina com política amostrada e é avaliado de forma gulosa. `stickystoch` mede a
  política amostrada.
- O ajuste de `K` e o desempate por quadros existem apenas nos GAs.
