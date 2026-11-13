# Guia do projeto: Problema das Marginais Quânticas

## Escopo e objetivo

Este repositório investiga o **Quantum Marginal Problem (QMP)** para sistemas multipartidos. Dadas matrizes densidade reduzidas em certos subconjuntos de subsistemas, pergunta-se se existe uma matriz densidade global `rho` cujas reduções parciais coincidam com todas elas.

Para dimensões locais `dims = (d_0, ..., d_{N-1})`, o espaço global tem dimensão `D = produto(dims)` e a variável global é uma matriz `D x D`. A formulação implementada é de factibilidade:

```text
encontrar rho >= 0
sujeito a Tr(rho) = 1 e Tr_{S^c}(rho) = omega_S para cada marginal prescrita S
```

As igualdades matriciais são codificadas em coordenadas de uma base Hermitiana ortonormal, usando dualidade de Hilbert-Schmidt, e passadas ao SDPLab. A resolução usa o adaptador CVXPY e, por padrão, CLARABEL.

## Organização dos arquivos

| Arquivo | Papel |
| --- | --- |
| `modulos/` | Pacote reutilizável dividido em `estados.py`, `operadores.py`, `validacao.py`, `sdp.py` e `experimentos.py`; `__init__.py` expõe a API comum. |
| `funcoes_fundamentais.py` | Camada temporária de compatibilidade que reexporta `modulos`; novos códigos devem importar do pacote. |
| `exemplos.ipynb` | Introdução às primitivas e à formulação SDP. |
| `exemplos_propostos.ipynb` | Casos de referência: GHZ, marginais maximamente mistas e marginais de Bell incompatíveis. |
| `exemplos_mestura.ipynb` | Família `t Bell + (1-t) I/4`, varredura do parâmetro e estudo adicional de compatibilidade. |
| `exemplos_compatibilidade.ipynb` | Experimentos com marginais aleatórias, produto/correlacionadas e comparação entre consistência local e extensibilidade. |
| `exemplos_maldicao_dimencao.ipynb` | Benchmarks empíricos do crescimento computacional. |
| `README.md` | Apresentação resumida do problema. |
| `Rascunho.pdf`, `Sla-1.pdf`, `imagens/`, PNGs | Materiais de apoio/figuras; sua relação detalhada com os notebooks não está explicitada no README. |

## Núcleo Python

As funções mais relevantes, na ordem aproximada do fluxo matemático, são:

- `basis`, `ketbra`, `tensor`, `ghz`: vetores/estados e produtos tensoriais.
- `verifica_densidade` e `_verify_density_matrix`: diagnósticos de matriz densidade.
- `partial_trace`: reduz subsistemas via contração `numpy.einsum`.
- `base_hermitiana`: base de `Herm(d)` com `d^2` elementos.
- `embed_operator`: incorpora um operador local no espaço global com identidades nos fatores restantes.
- `parse_subsystem_key`: converte chaves como `'AB'` em índices `(0, 1)`.
- `build_qmp_sdp`: constrói as igualdades afins do SDP, incluindo a normalização do traço por padrão.
- `solve_qmp`: executa o solver e retorna status, estado primal e dados duais/problema.
- `verify_marginals`: calcula erros de reconstrução, traço, hermiticidade, positividade e pureza.
- `random_density_matrix`, `build_product_marginals`, `build_correlated_marginals`: geração de exemplos para experimentos.
- `check_overlap_consistency`, `print_experiment_stats`: análise dos ensaios de compatibilidade.
- `DualEntropico`, `otimizar_dual`: formulação/avaliação do dual entrópico e minimização de `-D` por L-BFGS-B.
- `construir_restricoes_marginais`: expansão de marginais em observáveis globais, com remoção de dependências lineares compatíveis.

As funções estão agrupadas por responsabilidade no pacote `modulos`. A função `_verify_density_matrix` é mantida para os notebooks legados, mas delega a `verifica_densidade` para evitar duplicação dos critérios de validade.

## Convenções matemáticas e de índices

- Para três qubits, `A`, `B`, `C` correspondem aos índices `0`, `1`, `2`.
- A ordem dos fatores é a ordem do tensor/Kronecker: `A ⊗ B ⊗ C`. Portanto, a ordem interna de índices de uma matriz marginal precisa concordar com a sequência em sua chave.
- `partial_trace(rho, keep=...)` recebe subsistemas que permanecem; alternativamente, `trace_out=...` informa os descartados. Não se deve passar ambos.
- `dims` descreve o sistema global, não a marginal. Quando omitido, algumas rotinas inferem qubits a partir da dimensão total; para sistemas não qubit ou subconjuntos não triviais, informe `dims` explicitamente.
- Uma matriz densidade é Hermitiana, PSD e tem traço um. A implementação usa aritmética complexa e tolerâncias para arredondamento numérico.
- O SDP implementa restrições de observáveis `Tr(M_i rho) = b_i`; a transposição usada ao construir `DenseConstraintOp` é parte da convenção de pareamento da SDPLab. Não altere sem validar a identidade com exemplos conhecidos.

## Conclusões científicas registradas

As descrições e células documentais dos notebooks registram os seguintes resultados; devem ser tratados como conclusões do estudo atual, não como uma auditoria independente de todas as saídas:

1. **Sobreposição consistente não basta.** Se marginais compartilham subsistemas, as reduções sobrepostas precisam coincidir. Essa condição é necessária, mas casos como dois pares `AB` e `AC` maximamente emaranhados em Bell são incompatíveis por monogamia do emaranhamento.
2. **Marginais de GHZ são extensíveis.** As marginais bipartidas do estado GHZ têm extensão global pura, o próprio estado GHZ.
3. **Marginais maximamente mistas admitem extensão mista.** `I_4/4` em cada par é compatível com `I_8/8`. O notebook registra que não há extensão global pura com todas essas marginais simultaneamente.
4. **Mistura de Bell.** O notebook propõe o limiar `t = 2/3` para a extensão simultânea das marginais `AB` e `AC` da família `rho(t) = t |Phi+><Phi+| + (1-t) I_4/4`. A varredura numérica procura confirmar a fronteira; confira os resultados executados e a discretização antes de citar precisão ou igualdade exata.
5. **Escalabilidade.** Para subsistemas de dimensão local `d`, a dimensão global cresce como `d^N`, enquanto o número de entradas da matriz global cresce como `d^(2N)`. O notebook apresenta benchmarks até `N=6` qubits e até `d=6` com três partes; extrapolações de tempo dependem do hardware, solver e formulação.
6. **Experimentos aleatórios.** Os notebooks comparam marginais independentes, produto e correlacionadas. Resultados estatísticos dependem das sementes, do número de amostras e da construção dos ensembles; não os generalize sem registrar essas condições.

## Cuidados para alterações e interpretação

- Um status `optimal`/`optimal_inaccurate` é evidência numérica de factibilidade, não certificado simbólico. Reconstrua as marginais e inspecione erros, menor autovalor e traço.
- O código de solver captura `ValueError`; outros tipos de exceção podem escapar. Ao trabalhar em robustez, examine a API/versão real do solver e reporte falhas sem as classificar automaticamente como infactibilidade.
- `parse_subsystem_key` aceita letras e sequências de índices; confira índices repetidos, vazios ou fora da faixa ao estender o suporte. O parâmetro `num_qubits` existe, mas não determina o mapa atual.
- A função `mestura` realiza combinação convexa algébrica sem validar `t` ou os operandos. Valide entradas quando a garantia física for necessária.
- `build_correlated_marginals` é um gerador experimental. Não suponha que toda construção aleatória seja genuinamente correlacionada ou que `num_kraus` assegure uma propriedade sem conferir o corpo completo da função e os diagnósticos produzidos.
- Células de notebook podem conter instalação de pacotes, execuções longas e geração de figuras. Inspecione antes de executar; preserve sementes, parâmetros e resultados relevantes para reprodutibilidade.
- As dependências identificadas nos notebooks incluem NumPy, Pandas, Matplotlib, SpaceCore (em alguns trechos fixado em `0.4.2`), SDPLab, CVXPY e CLARABEL. O projeto não inclui manifesto de dependências nesta listagem observada; confirme o ambiente antes de assumir versões instaladas.

## Roteiro de leitura

1. Este guia e `AGENTS.md`.
2. `README.md` para a síntese do escopo.
3. `modulos/` para semântica real da implementação.
4. `exemplos.ipynb` para as primitivas e a convenção do SDP.
5. O notebook específico do fenômeno em estudo; consulte `exemplos_propostos.ipynb` para benchmarks compactos e os demais para experimentos.
