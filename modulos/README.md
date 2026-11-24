# Módulos do pacote `modulos`

Este diretório contém a implementação reutilizável para estados quânticos multipartidos, marginais, formulações de otimização e diagnósticos. O código é a referência para a semântica descrita aqui. `funcoes_fundamentais.py` reexporta os nomes de `modulos` por compatibilidade com consumidores antigos.

## Mapa dos arquivos

| Módulo | Responsabilidade | Símbolos principais |
|---|---|---|
| `estados.py` | Estados computacionais, projetores, produtos tensoriais, GHZ e mistura algébrica. | `zero`, `one`, `basis`, `ketbra`, `tensor`, `ghz`, `mestura` |
| `operadores.py` | Redução parcial, base de operadores hermitianos, embedding e interpretação de chaves. | `partial_trace`, `base_hermitiana`, `embed_operator`, `parse_subsystem_key` |
| `validacoes.py` | Validação rigorosa de dimensões, subsistemas, matrizes densidade, dados do dual e parâmetros de otimização. | `validar_escalar_positivo`, `validar_vetor_real`, `validar_hermitiana`, `validar_densidade`, `validar_dimensoes`, `validar_subsistema`, `validar_marginais`, `validar_dados_dual`, `validar_parametros_otimizacao` |
| `validacao.py` | Diagnósticos numéricos de estados e de marginais reconstruídas. | `verifica_densidade`, `_verify_density_matrix`, `verify_marginals` |
| `marginais.py` | Codificação de marginais prescritas como igualdades lineares globais. | `restricoes_lineares` |
| `sdp.py` | Construção e resolução do SDP de extensão global. | `build_qmp_sdp`, `solve_qmp` |
| `dual_entropico.py` | Estado exponencial, funcional dual entrópico, gradiente e resíduos. | `DualEntropico` |
| `otimizacao.py` | Algoritmos para maximizar o dual entrópico. | `otimizar_dual`, `gradient_acendent`, `gradient_acendent_sthocastical` |
| `experimentos.py` | Geração de estados/marginais aleatórios e resumo de experimentos de compatibilidade. | `random_density_matrix`, `build_product_marginals`, `build_correlated_marginals`, `check_overlap_consistency`, `print_experiment_stats` |
| `__init__.py` | Fachada de imports sob demanda. | nomes listados em `__all__` |

## Convenções

- A ordem dos sistemas é a ordem dos fatores no produto tensorial. Em três qubits, `A=0`, `B=1`, `C=2`; uma matriz para `AB` usa essa ordem. `dims` lista as dimensões locais na mesma ordem.
- `partial_trace(rho, keep=..., dims=...)` mantém os índices indicados; alternativamente, use `trace_out`. Informe exatamente um dos dois argumentos. Se `dims` for omitido, algumas funções inferem um sistema de qubits.
- `base_hermitiana(d)` retorna `d²` matrizes ortonormais no produto de Hilbert–Schmidt. `embed_operator(O, sistema, dims=...)` insere identidades nos fatores que não pertencem ao subsistema.
- As funções em `validacoes.py` rejeitam entradas inválidas com `ValueError` e retornam cópias numéricas. Validar cada marginal localmente não demonstra que exista uma extensão global.
- Os diagnósticos em `validacao.py` informam erros e propriedades numéricas. Tolerâncias são critérios computacionais, não certificados exatos.

## Fluxos de uso

### Marginais e SDP

`restricoes_lineares(marginals_dict, dims)` retorna `(M, b)`, representando as igualdades `Tr(M[i] @ rho) = b[i]`; inclui a normalização do traço e valida as marginais. Para a interface SDPLab, `build_qmp_sdp(marginals_dict, dims=None, include_trace_norm=True, ctx=None)` retorna o problema e os arrays `(M, b)`. `solve_qmp(sdp, solver="CLARABEL", ...)` retorna um dicionário com `status`, `is_feasible`, `rho`, `dual_y`, `problem` e `error_message`.

O status do solver é um resultado numérico. Quando `rho` estiver disponível, use `verify_marginals` para examinar traço, hermiticidade, menor autovalor e erro de cada redução. `verify_marginals` retorna `is_feasible=True` quando uma matriz foi fornecida; esse campo não substitui a inspeção dos outros diagnósticos.

### Dual entrópico

`DualEntropico(H, Q, q, epsilon)` recebe um Hamiltoniano hermitiano `H`, uma sequência de observáveis globais hermitianos `Q`, alvos reais `q` e regularização positiva `epsilon`. `estado(u)` calcula

$$\Gamma_u = \exp\!\left((\sum_i u_i Q_i-H)/\epsilon\right).$$

O estado retornado é não normalizado. `avaliar(u)` retorna o valor dual e o gradiente `q_i - Tr(Q_i Gamma_u)`; `residuos(u)` retorna o sinal oposto, `Tr(Q_i Gamma_u)-q_i`. Opcionalmente, `avaliar(u, mask=...)` calcula o gradiente somente para os índices selecionados, embora ainda avalie o estado completo.

`otimizar_dual` usa L-BFGS-B para minimizar o negativo do dual e retorna multiplicadores, estado, valor, resíduos, indicação de convergência e o resultado original do SciPy. `gradient_acendent` e `gradient_acendent_sthocastical` implementam alternativas por subida de gradiente; a segunda escolhe aleatoriamente coordenadas em cada passo. A convergência do algoritmo e a magnitude dos resíduos devem ser avaliadas separadamente.

### Experimentos

`random_density_matrix(d, rng=None)` gera uma matriz densidade aleatória (o gerador opcional permite controlar a aleatoriedade). `build_product_marginals` cria marginais de produto; `build_correlated_marginals` constrói exemplos correlacionados. `check_overlap_consistency` verifica consistência entre marginais sobrepostas e `print_experiment_stats` resume um DataFrame de resultados. Esses geradores apoiam experimentos e não provam afirmações gerais sobre compatibilidade.

## Imports

Para reduzir dependências incidentais, prefira importar do submódulo que contém a função:

```python
from modulos.operadores import partial_trace
from modulos.marginais import restricoes_lineares
from modulos.validacao import verify_marginals
from modulos.sdp import build_qmp_sdp, solve_qmp
```

O pacote também oferece uma fachada com carregamento sob demanda, por exemplo `from modulos import partial_trace`. O módulo `sdp` depende de `spacecore`, `sdplab` e do adaptador CVXPY; as primitivas de `operadores.py` não precisam carregar essas dependências. As rotinas de otimização usam SciPy; experimentos usam Pandas.

**Nota sobre a fachada:** no estado atual do código, `_EXPORTS` em `__init__.py` registra `RestricoesMarginais` e `construir_restricoes_marginais`, enquanto `marginais.py` define `restricoes_lineares`; essa correspondência precisa ser reconciliada antes de usar aqueles dois nomes pela fachada. Este guia recomenda o import direto mostrado acima. Também há alegações sobre uma API funcional de três funções em documentos locais que divergem do `dual_entropico.py` atual, cuja implementação é a classe `DualEntropico`.

## Limites e manutenção

O pacote cobre implementação numérica, não uma prova matemática geral. A ordenação dos tensores, o mapeamento das chaves, o pareamento usado pelo SDPLab e os diagnósticos de factibilidade são pontos sensíveis. Ao alterar essas convenções, confira `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` e os consumidores em notebooks/testes. O manifesto de dependências e a execução integral dos notebooks não estão estabelecidos por esta documentação.
