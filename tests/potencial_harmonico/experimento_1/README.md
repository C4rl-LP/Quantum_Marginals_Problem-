# Experimento 1

O experimento reconstrói o estado fundamental de dois osciladores harmônicos
acoplados a partir de suas matrizes densidade reduzidas. A configuração comum
fica em [base.py](base.py). Por padrão, cada modo tem dimensão 4, a frequência
é 1 e o acoplamento é 0,5.

Execute os comandos abaixo a partir da raiz do repositório, com as dependências
instaladas. Também é possível usar `python -m tests.potencial_harmonico.experimento_1.NOME_DO_SCRIPT`,
sem a extensão `.py`.

## Scripts executáveis

Todos os pontos de entrada `main(...)` retornam `None`. Os resultados aparecem
no terminal, em figuras ou nos arquivos descritos abaixo. Os caminhos das saídas
são relativos à pasta deste experimento.

### [Experimento_1.py](Experimento_1.py)

```bash
python tests/potencial_harmonico/experimento_1/Experimento_1.py
```

Usa o dual exponencial. Os valores de ε e o ε da figura de posições são
definidos em `main()` e ambos valem 0,1 por padrão. Começa com multiplicadores
nulos e, quando há vários valores de ε, usa os multiplicadores do passo anterior.

- Imprime energia, erro absoluto de energia, entropia, distância de Frobenius ao
  estado fundamental, norma dos resíduos, convergência, covariância e Pearson
  das posições.
- Acrescenta a configuração, as matrizes de referência, os resultados do
  otimizador e as marginais reconstruídas a um arquivo em `protocolos/`. A
  assinatura da configuração fixa determina o arquivo usado em cada execução.
- Exibe as métricas em função de ε, as marginais prescritas e as marginais da
  última reconstrução. Exibe também a densidade conjunta de posições se o ε
  selecionado para visualização fizer parte da varredura.
- `iniciar_protocolo(...)` retorna o caminho do arquivo. `registrar_resultado(...)`
  acrescenta um bloco ao protocolo e retorna `None`.

Para gerar a figura de Frobenius para os slides:

```bash
python tests/potencial_harmonico/experimento_1/Experimento_1.py --figura-frobenius
```

A opção usa o dual logarítmico em 21 valores de ε entre 0,01 e 1, com
multiplicadores inicialmente nulos em cada ponto. Grava
`figuras_slides/distancia_frobenius.png` e imprime o caminho retornado por
`figura_frobenius()`. A distância numérica aparece junto das curvas
`sqrt(2 ε [S(ρ_A) + S(ρ_B)] / ω)` e
`sqrt(2 ε [S(ρ_A) + S(ρ_B)] / Δ)`, calculadas com as marginais prescritas.
O gap espectral Δ é a diferença entre os dois menores autovalores do
Hamiltoniano acoplado. A curva com ω usa a frequência local como referência.
O símbolo × marca reconstruções fora da tolerância do otimizador.

### [experimento_learnrate.py](experimento_learnrate.py)

```bash
python tests/potencial_harmonico/experimento_1/experimento_learnrate.py --forma logaritmico
```

Aceita `--forma exponencial` (padrão), `--forma logaritmico` e o alias
`--forma log`. Reutiliza a varredura de taxas do experimento de marginais
térmicas com as marginais do estado fundamental fixas. Compara sete fatores
do passo de gradiente em cinco valores de ε entre 0,01 e 0,1. Imprime um resumo
de tolerância por fator e grava `erro_por_learnrate_<forma>.png` com a norma
do resíduo final de cada combinação.

### [correlacoes_por_acoplamento.py](correlacoes_por_acoplamento.py)

```bash
python tests/potencial_harmonico/experimento_1/correlacoes_por_acoplamento.py
```

Reconstrói os estados com o dual logarítmico para cinco acoplamentos
(g = 0, 0,25, 0,5, 0,75 e 0,9) e 21 valores de ε de 0,01 a 1. Compara
a covariância e o coeficiente de Pearson das posições com os valores do
estado fundamental de cada acoplamento. Imprime o resumo dos resíduos e
de Pearson por acoplamento. Grava `figuras_slides/correlacoes_por_acoplamento.png`
e marca com × os pontos fora da tolerância.

### [wigner_global.py](wigner_global.py)

```bash
python tests/potencial_harmonico/experimento_1/wigner_global.py --epsilon 0.1
```

Reconstrói um estado com o dual logarítmico para o ε informado
(padrão 100). Compara o estado fundamental, a reconstrução e a diferença
entre eles em dois cortes da Wigner conjunta. Imprime o resíduo final e a
diferença absoluta máxima de cada corte. Grava
`figuras_slides/wigner_global_epsilon_<epsilon>.png`, com ε formatado
sem zeros decimais extras.

A função `cortes_wigner(rho, grade, keep, dims=None)`, em
[modulos/visualizacao.py](../../../modulos/visualizacao.py), retorna as matrizes
2D `(posicoes, momentos)` para os dois modos selecionados em `keep`. A primeira
fixa os momentos em zero e a segunda fixa as posições em zero. Em cada matriz,
as colunas correspondem ao primeiro modo e as linhas ao segundo, na ordem de
`dims`.

## Módulos de apoio

- [base.py](base.py) concentra os parâmetros físicos e os do otimizador.
  `ExperimentoHarmonico(g)` disponibiliza o Hamiltoniano, a energia e o estado
  fundamental, as marginais prescritas e as restrições lineares `Q, q`.
  `calcular_marginais(estado)` retorna as matrizes densidade reduzidas
  indexadas por `(modo,)`. `momentos_posicao(estado)` retorna o dicionário de
  momentos de posição, incluindo covariância e Pearson. `criar_problema(epsilon,
  forma)` retorna um `DualEntropico`. `reconstruir(epsilon, forma, u_inicial=None)`
  retorna o dicionário do otimizador, com estado reconstruído, multiplicadores,
  valor dual, gradiente, resíduos, indicador de convergência e histórico. O
  passo padrão é 0,1ε, a tolerância é 10⁻⁶ e o limite é de 10.000 iterações.
  Se nenhum vetor inicial for fornecido, o otimizador começa em zero.
- [figuras.py](figuras.py) fornece `salvar(figura, nome)`. A função cria
  `figuras_slides/` quando necessário, grava o PNG a 200 dpi, fecha a figura e
  retorna o caminho salvo.
