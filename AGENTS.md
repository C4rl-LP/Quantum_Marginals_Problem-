# Orientação para agentes de IA

Leia este arquivo antes de alterar o projeto. Para contexto científico e mapa dos materiais, consulte também [GUIA_DO_PROJETO.md](GUIA_DO_PROJETO.md).

## Objetivo

O projeto estuda o Problema das Marginais Quânticas (QMP): decidir numericamente se matrizes densidade reduzidas prescritas admitem uma extensão global positiva semidefinida, com traço um. A formulação principal é um problema de factibilidade SDP resolvido via SDPLab/CVXPY e CLARABEL.

## Mapa

- `funcoes_fundamentais.py`: implementação reutilizável das operações quânticas, montagem/resolução do SDP, verificação e geradores experimentais.
- `exemplos.ipynb`: fundamentos e demonstração das primitivas.
- `exemplos_propostos.ipynb`: benchmarks GHZ, marginais maximamente mistas e incompatibilidade de Bell.
- `exemplos_mestura.ipynb`: varredura da mistura de Bell com estado maximamente misto e estudo de compatibilidade.
- `exemplos_compatibilidade.ipynb`: experimentos estatísticos sobre consistência de sobreposição e extensibilidade global.
- `exemplos_maldicao_dimencao.ipynb`: benchmarks de escalabilidade em número de subsistemas e dimensão local.
- `README.md`: resumo curto do escopo.
- PDFs e imagens: materiais de apoio e saídas visuais já presentes no repositório; consulte-os quando a tarefa depender deles.

## Regras e convenções de trabalho

1. Leia `GUIA_DO_PROJETO.md` e os trechos pertinentes dos notebooks antes de modificar a implementação. Notebooks são parte da documentação científica, não apenas demonstrações descartáveis.
2. A ordem dos subsistemas é a ordem dos fatores do produto tensorial: `A=0`, `B=1`, `C=2`; `np.kron` e as chaves `'AB'`, `'AC'` seguem essa ordem.
3. Estados e marginais são matrizes complexas; estados válidos devem ser Hermitianos, PSD e de traço unitário. Use tolerâncias numéricas explícitas ao interpretar resultados do solver.
4. O SDP busca `rho >= 0` com igualdade de traço e observáveis locais que codificam as marginais. Compatibilidade de marginais sobrepostas é necessária, mas em geral não suficiente para extensão global.
5. Não descreva uma conclusão empírica ou anotada em notebook como prova geral. Diferencie teoremas, expectativas, resultados numéricos e limitações do solver.
6. Preserve a API existente de `funcoes_fundamentais.py` quando possível. Mudanças em `partial_trace`, `embed_operator`, `parse_subsystem_key` ou `build_qmp_sdp` exigem atenção especial à ordenação e às dimensões locais.
7. Os notebooks mencionam instalações de dependências em células. Não execute essas células automaticamente; primeiro inspecione o ambiente e a necessidade da tarefa.
8. Ao reportar factibilidade, considere validar a solução reconstruindo as marginais e verificando traço, hermiticidade e espectro. Um status do solver, isoladamente, não substitui o diagnóstico numérico.

## Estado conhecido do repositório

O código central contém funções para base, projetores, tensor, GHZ, traço parcial, embedding, formulação/resolução SDP, validação, geração de estados de Ginibre e construção de exemplos de marginais. Os notebooks registram conclusões e experimentos descritos no guia. Não presuma que todos os notebooks foram executados no ambiente atual sem verificação.

## Documentação modular

- Para retomar trabalho, comece por `docs/CURRENT_STATE.md`; para componentes, `docs/ARCHITECTURE.md`; para escolhas, `docs/DECISIONS.md`.
- O guia e o código são fontes primárias. Resultados narrados em notebooks não devem ser promovidos a teoremas nem a resultados revalidados sem inspecionar saídas, parâmetros e diagnósticos.
- Atualize primeiro documentação local afetada após mudanças relevantes. Resumo central: `~/AI_CONTEXT/projetos/Quantum_marginal_problem.md`.
