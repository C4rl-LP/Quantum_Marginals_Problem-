# Problema das Marginais Quânticas (QMP)

Este repositório estuda computacionalmente o **Quantum Marginal Problem**: dadas matrizes densidade reduzidas em subsistemas especificados, determinar se existe um estado global que as tenha como marginais. O núcleo formula a pergunta como um problema de factibilidade por programação semidefinida (SDP).

Para subsistemas de dimensões locais `dims = (d₀,…,dₙ₋₁)`, procura-se uma matriz `ρ` no espaço global de dimensão `D = ∏ᵢ dᵢ` tal que

\[
ρ \succeq 0,\qquad \mathrm{Tr}(ρ)=1,\qquad \mathrm{Tr}_{S^c}(ρ)=ω_S
\]

para cada marginal prescrita `ω_S`. As igualdades são codificadas em uma base Hermitiana e resolvidas pela integração SDPLab/CVXPY; o guia identifica CLARABEL como solver padrão. Compatibilidade entre marginais sobrepostas é necessária, mas não garante, em geral, uma extensão global.

## Organização

- `funcoes_fundamentais.py`: estados, traço parcial, embedding local, formulação/resolução do SDP, validação e geradores experimentais.
- `exemplos.ipynb`: primitivas e formulação.
- `exemplos_propostos.ipynb`: exemplos de GHZ, marginais maximamente mistas e incompatibilidade de Bell.
- `exemplos_mestura.ipynb`: família de misturas de Bell e varreduras numéricas.
- `exemplos_compatibilidade.ipynb`: experimentos sobre consistência de sobreposição e extensibilidade.
- `exemplos_maldicao_dimencao.ipynb`: benchmarks de escala.
- `GUIA_DO_PROJETO.md`: referência detalhada de escopo, funções, convenções e cautelas.
- `imagens/`, PNGs e PDFs: materiais visuais/de apoio; a proveniência de cada saída deve ser consultada antes de reutilização.

## Requisitos e execução

O código importa Python, NumPy, Pandas, Matplotlib, SpaceCore, SDPLab e usa CVXPY/CLARABEL por meio do adaptador indicado no guia. Não há manifesto de dependências do projeto. Algumas células dos notebooks contêm comandos de instalação, mas não são um procedimento de instalação verificado e não devem ser executadas automaticamente. Configure as dependências em ambiente apropriado e abra o notebook pertinente; não foi validada nesta documentação uma sequência reproduzível de execução ponta a ponta.

## Resultados e limitações

Os notebooks documentam benchmarks analíticos e experimentais: extensões de marginais de GHZ e maximamente mistas, incompatibilidade de marginais de Bell por monogamia, e estudos de mistura/compatibilidade/escalabilidade. São registros de experimentos, não uma validação independente nesta revisão. O limiar `t=2/3` para uma família de mistura é proposto nos materiais e sua precisão deve ser conferida contra a discretização e as execuções. Tempos dependem de hardware, solver e formulação. Status numérico do solver não substitui reconstrução das marginais e checagem de positividade, traço e hermiticidade.

## Documentação

Leia primeiro [AGENTS.md](AGENTS.md), depois [arquitetura](docs/ARCHITECTURE.md), [decisões](docs/DECISIONS.md) e [estado atual](docs/CURRENT_STATE.md). As afirmações técnicas detalhadas e as convenções de índices estão em `GUIA_DO_PROJETO.md` e devem ser comparadas com o código quando a tarefa tocar a implementação.
