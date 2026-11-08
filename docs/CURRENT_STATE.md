# Estado atual

Última verificação: 2026-11-07.

## Objetivo

Investigar computacionalmente a existência de extensões globais para conjuntos de marginais quânticas, com ênfase em sistemas multipartidos e exemplos/experimentos numéricos.

## Implementação

**Verificado por inspeção:** `funcoes_fundamentais.py` contém funções para estados, traço parcial, embedding, parsing de subsistemas, montagem/resolução SDP, diagnóstico de marginais e geração de ensembles/exemplos. O módulo importa bibliotecas externas no topo do arquivo.

**Documentado:** os notebooks estudam exemplos de GHZ, marginais maximamente mistas, incompatibilidade de Bell, mistura de Bell, ensembles aleatórios e escalabilidade. Os notebooks incluem células de instalação e narrativa de resultados; as execuções não foram revalidadas nesta tarefa.

**Não verificado:** ambiente reproduzível, versões efetivamente instaladas, execução integral dos notebooks e proveniência de todas as figuras/PDFs.

## Testes e problemas conhecidos

Não há suíte automatizada convencional identificada. Os notebooks são experimentos, não testes de regressão garantidos. O guia chama atenção para verificar solver e precisão, reconstruir marginais e medir traço, hermiticidade e menor autovalor. Há divergência nos nomes de dependências em células (`sdplab`/`sdp`) que deve ser investigada antes de instruções de instalação. Nenhum experimento foi executado nesta verificação.

## Pendências e próximos passos

1. Estabelecer um manifesto de dependências reproduzível sem executar as células de instalação dos notebooks às cegas.
2. Separar testes determinísticos de regressão dos experimentos extensos em notebooks.
3. Registrar para cada figura/resultado parâmetros, sementes e origem da execução.
4. Revalidar alegações experimentais, inclusive o limiar proposto `t=2/3`, com grade/resolução adequadas e diagnósticos do estado global.
5. Revisar validação de entradas e tratamento de exceções conforme os cuidados do guia antes de ampliar o domínio de dimensões/subsistemas.

## Arquivos para continuar

- `AGENTS.md` e `GUIA_DO_PROJETO.md` para regras e convenções.
- `funcoes_fundamentais.py` para semântica implementada.
- `exemplos_propostos.ipynb` para benchmarks de referência.
- Notebook específico para mistura, compatibilidade ou escalabilidade.
