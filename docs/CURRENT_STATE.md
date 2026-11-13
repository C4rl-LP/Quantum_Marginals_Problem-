# Estado atual

Última atualização: 2026-11-12.

## Objetivo

Investigar computacionalmente a existência de extensões globais para conjuntos de marginais quânticas, com ênfase em sistemas multipartidos e exemplos/experimentos numéricos.

## Implementação

**Verificado por inspeção:** `modulos/` separa estados, operadores, validação, SDP e utilidades experimentais. `funcoes_fundamentais.py` reexporta a API como compatibilidade para consumidores antigos. A validação antiga `_verify_density_matrix` delega ao diagnóstico principal, mantendo o dicionário legado.

O pacote inclui `DualEntropico` em `modulos/dual_entropico.py`, com avaliação conjunta do dual entrópico, gradiente, estado não normalizado e resíduos. As entradas de operadores são verificadas quanto à dimensão, finitude e hermiticidade.

`modulos/marginais.py` gera restrições globais e elimina dependências lineares consistentes. `modulos/otimizacao.py` aplica L-BFGS-B e retorna separadamente convergência do algoritmo e resíduos. Testes automatizados em `tests/` cobrem gradiente, reconstrução de produto e benchmark analítico de Ising; marginais singulares documentam o possível não atingimento do supremo em parâmetros finitos.

**Documentado:** os notebooks estudam exemplos de GHZ, marginais maximamente mistas, incompatibilidade de Bell, mistura de Bell, ensembles aleatórios e escalabilidade. Os notebooks incluem células de instalação e narrativa de resultados; as execuções não foram revalidadas nesta tarefa.

**Verificado nesta alteração:** sintaxe dos módulos (análise AST), JSON dos notebooks e migração das importações dos cinco notebooks dependentes para `modulos`.

**Não verificado:** ambiente reproduzível, versões efetivamente instaladas, execução integral dos notebooks e proveniência de todas as figuras/PDFs.

## Testes e problemas conhecidos

Há testes automatizados convencionais em `tests/test_dual_entropico.py`, executados com `python -m unittest discover -s tests -v`. Eles cobrem o gradiente, positividade do estado exponencial, marginais produto/sobrepostas, o caso analítico de Ising, redundâncias, marginais singulares e overflow. Os notebooks científicos não foram executados integralmente. Há divergência nos nomes de dependências em células (`sdplab`/`sdp`) que deve ser investigada antes de instruções de instalação.

## Pendências e próximos passos

1. Estabelecer um manifesto de dependências reproduzível sem executar as células de instalação dos notebooks às cegas.
2. Registrar para cada figura/resultado parâmetros, sementes e origem da execução.
3. Revalidar alegações experimentais, inclusive o limiar proposto `t=2/3`, com grade/resolução adequadas e diagnósticos do estado global.
4. Revisar validação de entradas e tratamento de exceções conforme os cuidados do guia antes de ampliar o domínio de dimensões/subsistemas.

## Arquivos para continuar

- `AGENTS.md` e `GUIA_DO_PROJETO.md` para regras e convenções.
- `modulos/` para semântica implementada.
- `exemplos_propostos.ipynb` para benchmarks de referência.
- Notebook específico para mistura, compatibilidade ou escalabilidade.
