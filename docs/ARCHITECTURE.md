# Arquitetura

O projeto reúne um módulo Python reutilizável e notebooks científicos. O fluxo conceitual é:

```mermaid
flowchart LR
  E[Estados e marginais prescritos] --> C[Convenções de subsistemas e traço parcial]
  C --> F[Restrições na base Hermitiana]
  F --> S[SDP / SDPLab e adaptador CVXPY]
  S --> R[Estado primal e status]
  R --> V[Reconstrução e diagnósticos]
  V --> N[Análises e figuras nos notebooks]
```

## Módulo principal

`funcoes_fundamentais.py` importa NumPy, Pandas, Matplotlib, SpaceCore e SDPLab no nível do módulo.

- `basis`, `ketbra`, `tensor`, `ghz`: vetores, projetores e estados de exemplo.
- `verifica_densidade` e `_verify_density_matrix`: diagnósticos de propriedades de matrizes densidade.
- `partial_trace`: reduz subsistemas por contração NumPy; aceita `keep` ou `trace_out` e dimensões `dims`.
- `base_hermitiana`: constrói a base usada na expansão de observáveis/marginais.
- `embed_operator`: incorpora um operador local no espaço tensorial global.
- `parse_subsystem_key`: interpreta chaves de subsistema, como `AB`, segundo a convenção de índices.
- `build_qmp_sdp`: monta restrições afins e normalização do traço no SDP.
- `solve_qmp`: chama o solver e retorna o estado/status e objetos associados.
- `verify_marginals`: compara marginais reconstruídas e calcula diagnósticos numéricos.
- `mestura`, `random_density_matrix`, `build_product_marginals`, `build_correlated_marginals`: utilidades e geradores experimentais.
- `check_overlap_consistency`, `print_experiment_stats`: análise estatística de experimentos.

## Convenções e dados

A ordem dos subsistemas acompanha os fatores do produto tensorial. Para três qubits, `A=0`, `B=1`, `C=2`; a ordem em `AB` e na matriz correspondente importa. `dims` descreve o sistema global. A formulação trabalha com `ρ ⪰ 0`, `Tr(ρ)=1` e igualdade entre reduções prescritas e marginais-alvo. O guia ressalta que transposição/pareamento de `DenseConstraintOp` depende da convenção SDPLab.

Os notebooks constroem estados/marginais, alimentam `build_qmp_sdp`/`solve_qmp`, e calculam tabelas/figuras. As saídas têm diferentes níveis de rastreabilidade; leia células e parâmetros antes de tratar números ou imagens como resultados reproduzíveis.

## Testes e extensões

Não há diretório de testes automatizados. Os notebooks funcionam como experimentos e demonstrações, mas não equivalem a uma suíte reproduzível. Pontos sensíveis de extensão: ordenação tensorial, dimensões locais não binárias, traço parcial, embedding, representação das restrições SDP e interpretação de status/tolerâncias.
