# Módulos do problema das marginais quânticas

O pacote `modulos` reúne operações sobre estados e Hamiltonianos, construção
das restrições marginais, otimização e visualização. Os scripts em `tests/`
escolhem os parâmetros físicos, os pontos das varreduras e os arquivos de saída.

## Mapa dos módulos

| Módulo | Responsabilidade | Funções principais |
| --- | --- | --- |
| `estados.py` | Construção de estados e cálculo de propriedades. | `basis`, `ketbra`, `tensor`, `ghz`, `mestura`, `estado_fundamental`, `estado_gibbs`, `estado_termico_oscilador`, `entropia_von_neumann`, `negatividade` |
| `Hamiltoniano_ex.py` | Hamiltonianos de osciladores e de cadeias abertas. | `hamiltoniano_oscilador`, `hamiltoniano_bilinear`, `Hamiltoniano_ZZ`, `Hamiltoniano_XY`, `dimensoes_qubits` |
| `operadores.py` | Operações locais e multipartidas. | `partial_trace`, `embed_operator`, `base_hermitiana`, `parse_subsystem_key`, `operador_posicao`, `momentos_observaveis_locais` |
| `marginais.py` | Reduções de estados e construção das igualdades marginais. | `reduzir_marginais`, `restricoes_lineares` |
| `validacao.py` | Validação de entradas, diagnóstico físico do estado e comparação de marginais. | `validar_densidade`, `validar_dimensoes`, `validar_marginais`, `validar_dados_dual`, `comparar_marginais`, `verify_marginals`, `verifica_densidade` |
| `sdp.py` | Formulação e resolução da factibilidade por SDP. | `build_qmp_sdp`, `solve_qmp` |
| `dual_entropico.py` | Formulações exponencial e logarítmica do dual. | `DualEntropico` |
| `otimizacao.py` | Maximização do dual e varreduras numéricas. | `otimizar_dual`, `gradient_acendent`, `gradient_acendent_sthocastical`, `varrer_dual`, `varrer_taxas_dual` |
| `diagnosticos_reconstrucao.py` | Estimativas espectrais e certificados de incompatibilidade. | `gap_espectral`, `cota_frobenius`, `avaliar_testemunha` |
| `visualizacao.py` | Wigner, densidade de posições, matrizes e gráficos de resíduos. | `wigner_marginal`, `cortes_wigner`, `visualizar_marginais`, `distribuicao_conjunta_posicoes`, `visualizar_posicoes_conjuntas`, `visualizar_matriz_densidade`, `visualizar_taxas_dual`, `salvar_figura` |
| `experimentos.py` | Geradores aleatórios e resumos de estudos de compatibilidade. | `random_density_matrix`, `build_product_marginals`, `build_correlated_marginals`, `check_overlap_consistency`, `print_experiment_stats` |

## Estados e marginais

`dims` segue a ordem dos fatores tensoriais. Em três qubits, A, B e C
correspondem aos índices 0, 1 e 2. `partial_trace` recebe exatamente um entre
`keep` e `trace_out`. A base de `base_hermitiana(d)` é ortonormal no produto
de Hilbert–Schmidt.

`estado_termico_oscilador(dimensao, temperatura, omega)` usa o Hamiltoniano
local de Fock truncado e retorna a matriz de Gibbs com temperatura inversa
(1/T), em unidades com (k_B=1). A temperatura local é independente do
parâmetro de regularização ε. `negatividade` mede a bipartição entre o
primeiro qubit e o restante do sistema.

As funções `validar_*` em `modulos.validacao` rejeitam entradas inválidas com
`ValueError`. `verifica_densidade` informa hermiticidade, positividade e traço
unitário com tolerância absoluta, o mesmo critério de `validar_densidade`.

`reduzir_marginais(estado, sistemas, dims)` retorna as matrizes reduzidas
indexadas pelas chaves fornecidas. `comparar_marginais(estado, marginais, dims)`
retorna, por chave, a redução, o alvo, a norma de Frobenius e o maior erro
absoluto de entrada. `verify_marginals` acrescenta o diagnóstico do estado
global, incluindo traço, espectro e pureza. Seu campo histórico
`is_feasible` informa se uma matriz foi fornecida.

## Otimização e diagnóstico

`restricoes_lineares(marginais, dims)` retorna `(Q, q)` para as igualdades
`Tr(Q[i] @ estado) = q[i]`, incluindo a normalização. A alternativa por SDP
usa `build_qmp_sdp` e `solve_qmp`, cujo resultado inclui status, estado e
multiplicadores quando disponíveis. A validação local das marginais e a
consistência no sistema compartilhado podem ser verificadas antes do SDP.

`DualEntropico(H, Q, q, epsilon, forma=...)` calcula o estado exponencial
ou, na forma `"logaritmico"`, o estado normalizado por Log-Sum-Exp.
`avaliar(u)` retorna o valor dual e o gradiente
`q[i] - Tr(Q[i] @ estado)`. `residuos(u)` retorna o sinal oposto.

`otimizar_dual` usa subida de gradiente, com os métodos `"gradient"` e
`"stochastic"`. Retorna multiplicadores, valor dual, gradiente, resíduos,
histórico e convergência pela tolerância. A matriz de estado é incluída por
padrão.

- `varrer_dual(criar_problema, epsilons, fator_learnrate=..., ...)` gera um
  resultado por ε, na ordem fornecida, inclusive sem convergência. O passo
  é `fator_learnrate * epsilon`. A inicialização pode ser um vetor, uma
  função do problema ou zero por padrão. Com
  `reutilizar_multiplicadores=True`, os pontos seguintes começam com a solução
  numérica do ponto anterior.
- `varrer_taxas_dual(criar_problema, epsilons, fatores, ...)` retorna uma
  matriz com os resíduos finais, com uma linha por fator e uma coluna por ε.
- `gap_espectral(H)` retorna a diferença entre os dois menores autovalores
  com multiplicidade. `cota_frobenius(epsilons, gap, marginais)` usa a soma
  das entropias prescritas. A interpretação como cota vale para a solução
  ótima factível com referência fundamental pura e gap positivo.
- `avaliar_testemunha(W, alvo)` retorna o maior autovalor de (W) e a
  diferença entre o alvo e esse máximo. Uma diferença positiva certifica
  que o alvo é inalcançável por um estado normalizado.

## Visualização e imports

`visualizar_posicoes_conjuntas` e `visualizar_taxas_dual` retornam a figura
e o eixo. `visualizar_matriz_densidade` desenha a parte real em um eixo
fornecido e aceita rótulos para a base. `salvar_figura(figura, caminho, dpi=...)`
cria o diretório de destino, grava e fecha a figura e retorna o caminho.

Prefira importar diretamente dos submódulos. O carregamento sob demanda de
`modulos/__init__.py` também oferece alguns nomes históricos, como
`partial_trace`. O módulo `sdp` usa SDPLab, SpaceCore e o adaptador CVXPY.
A visualização usa Matplotlib e SciPy e carrega QuTiP para a Wigner de um modo.
