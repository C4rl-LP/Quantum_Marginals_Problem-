# 2–3. Diagnóstico do código e matriz de reaproveitamento

## Fluxo efetivamente implementado

O código de biblioteca reside em [modulos](../../modulos), demonstrações em [exemplos](../../exemplos) e scripts científicos e testes em [tests](../../tests). Os notebooks estão na raiz. A auditoria leu o [README](../../README.md), a [arquitetura](../ARCHITECTURE.md), as [decisões](../DECISIONS.md), o [estado documentado](../CURRENT_STATE.md) e o [guia](../../imagens/GUIA_DO_PROJETO.md). A referência a AGENTS.md no README aponta para um arquivo ausente nas localizações aplicáveis inspecionadas. Rascunho.pdf e Sla-1.pdf contextualizam a formulação de marginais e SDP. A documentação histórica menciona duas APIs diferentes do dual, redução QR e L-BFGS-B. A implementação inspecionada é a classe com exponencial de SciPy, restrições completas e subida de gradiente.

1. O script harmônico monta H denso e chama restricoes_lineares. As marginais são um dict de letras ou tuplas de índices para matrizes densidade. Use dimensões explícitas, chaves (0,), (1,), e ordem tensorial conforme dims.
2. A expansão usa uma identidade global com alvo 1 e uma base hermitiana completa por marginal. Retorna M complexo com forma (m,D,D) e b real com forma (m,).
3. DualEntropico valida e copia H, Q e q. Para cada u, monta o operador efetivo e usa scipy.linalg.expm. avaliar retorna um escalar e o gradiente, com sinal \(q-\langle Q\rangle\). A derivada da função traço exponencial usa ciclicidade, inclusive quando os operadores não comutam.
4. otimizar_dual maximiza esse escalar. Depois avalia o gradiente completo e, por padrão, chama estado mais uma vez para produzir a matriz global.
5. verify_marginals recebe a matriz e devolve erros Frobenius e máximos de cada redução, traço, espectro, positividade e pureza. is_feasible significa que recebeu matriz (linhas 54 e 87), e precisa ser interpretado junto dos diagnósticos.

| Informação | Disponibilidade em otimizar_dual |
|---|---|
| Matriz global | estado, incluída se salvar_matriz_densidade=True |
| Multiplicadores, dual, gradiente | multiplicadores, valor_dual, gradiente |
| Resíduos | residuos, norma_residuos, residuo_maximo |
| Método, histórico | metodo, historico com pares (valor, norma) antes das atualizações. No estocástico, a maior parte do histórico usa apenas coordenadas selecionadas |
| Iterações, avaliações, motivo de parada | Campos ausentes. len(historico) conta entradas e pode incluir a avaliação terminal. No estocástico também há avaliações completas adicionais |
| Critério/indicador | convergiu compara a norma euclidiana final completa com tolerancia, com desigualdade estrita. O dicionário não registra motivo nem a tolerância aplicada |
| Energia, entropia, tempo, erros físicos | Calculados na camada experimental |

## Matriz de reaproveitamento

| Funcionalidade | Já existe? | Arquivo atual e assinatura | Pode ser reutilizada? | Alteração necessária |
|---|---|---|---|---|
| Hamiltoniano bilinear | Sim | [script harmônico, linha 17](../../tests/potencial_harmonico_sem_interacao.py#L17), hamiltoniano_bilinear(dims, omega, pares_interacao) | Sim | Extrair a implementação para camada física, adicionar validação e manter a convenção |
| Montagem harmônica do dual | Sim | [script harmônico, linha 52](../../tests/potencial_harmonico_sem_interacao.py#L52), construir_problema(marginals_dict, dims, omega=1.0, epsilon=1.0, pares_interacao=()) | Sim | Manter wrapper e usar caminho que permite H pré-calculado nas varreduras |
| Wigner individual | Sim | [script harmônico, linha 76](../../tests/potencial_harmonico_sem_interacao.py#L76), visualizar_marginais(marginals_dict, limite=5.0, pontos=101) → figura, eixos | Sim | Usar import opcional de QuTiP quando a figura for solicitada |
| Traço parcial | Sim | [operadores.py, linha 7](../../modulos/operadores.py#L7), partial_trace(rho, keep=None, trace_out=None, dims=None) | Sim | Passar dimensões e índices validados. O código ordena keep |
| Embedding e base | Sim | [operadores.py, linha 77](../../modulos/operadores.py#L77), base_hermitiana(d); [linha 105](../../modulos/operadores.py#L105), embed_operator(O_s, sistema, dim=None, *, dims=None) | Sim | Preservar ordem tensorial de sistema e da matriz |
| Chaves de subsistema | Sim | [operadores.py, linha 148](../../modulos/operadores.py#L148), parse_subsystem_key(key, num_qubits=3) | Para chaves já válidas | Usar validar_subsistema nas entradas. O parser ignora caracteres não alfabéticos, converte sequências sem validar limites e num_qubits não governa o mapa |
| Restrições lineares | Sim | [marginais.py, linha 6](../../modulos/marginais.py#L6), restricoes_lineares(marginals_dict, dims) → M,b | Sim | Para marginais individuais, omitir um projetor diagonal por modo sob opção explícita, preservando a identidade global |
| Dual entrópico | Sim | [dual_entropico.py, linha 6](../../modulos/dual_entropico.py#L6), DualEntropico(H,Q,q,epsilon), estado(u), avaliar(u,mask=None), residuos(u) | Sim | Corrigir máscara se houver uso estocástico. Reusar convenção e converter o valor na análise |
| Subida de gradiente | Sim | [otimizacao.py, linha 8](../../modulos/otimizacao.py#L8), otimizar_dual(problema,u_inicial=None,tolerancia=1e-8,maxiter=2000,metodo="gradient",salvar_matriz_densidade=True,**kwargs) | Sim | Medir desempenho. Busca linear e contadores explícitos após referência inicial |
| Diagnóstico de densidade e marginais | Sim | [validacao.py, linha 8](../../modulos/validacao.py#L8), verifica_densidade(rho,tol=1e-10); [linha 44](../../modulos/validacao.py#L44), verify_marginals(rho,marginals_dict,dims=None,tol=1e-7) | Sim | Manter campos. Aplicar limites físicos próprios e chaves unitárias canônicas |
| Validação de entradas | Sim | [validacoes.py](../../modulos/validacoes.py), validar_dimensoes, validar_densidade, validar_subsistema, validar_marginais, validar_dados_dual e validar_parametros_otimizacao | Sim | Acrescentar só validações ligadas ao domínio de osciladores |
| Produto, projetor e mistura | Sim | [estados.py, linha 16](../../modulos/estados.py#L16), ketbra(psi), tensor(*args), mestura(rho_1,rho_2,t,dim=None) | Sim | Ordenar marginais por índice e validar p. build_product_marginals gera exemplos aleatórios |
| Estados de osciladores | Ausentes como API reutilizável | [estados.py](../../modulos/estados.py), [experimentos.py](../../modulos/experimentos.py) | Primitivas sim | Adicionar primeiro estado térmico e controles mínimos |
| Entropia e negatividade | Sim, locais aos scripts | [Experimento_1.py, linhas 52–65](../../tests/Experimento_1.py#L52), entropia_von_neumann(estado), negatividade(estado); [cópia, linhas 72–86](../../tests/Experimento_1_copia.py#L72) | Fórmulas sim | Promover entropia espectral e generalizar transposição por dims e bipartição. Pureza já está em verify_marginals |
| SDP | Factibilidade | [sdp.py, linhas 11 e 81](../../modulos/sdp.py#L11), build_qmp_sdp(marginals_dict,dims=None,include_trace_norm=True,ctx=None), solve_qmp(sdp,solver="CLARABEL",verbose=False,**kwargs) | Como referência estrutural | Custo é zero. Benchmark energético e restrição PPT exigem formulação própria pequena com dependências opcionais |
| Busca restrita energética e minimização externa | Ausentes | — | Peças acima | Orquestrar por funções |

## Problemas e sondagens executadas

- **Redundância.** Para dims=(2,3), restricoes_lineares produziu (14,6,6), de posto 12. Para marginais unitárias, a soma das diagonais de cada modo reproduz I global. Isso cria direções planas e impede inversão ingênua da Hessiana. Em norma absoluta os resíduos também dependem da base escolhida.
- **Máscara.** [dual_entropico.py, linha 27](../../modulos/dual_entropico.py#L27) usa mask == None. A chamada estocástica com dim_choice=2 falhou por ambiguidade do array. dim_choice=1 percorreu duas iterações na sondagem. A exponencial é completa nos dois métodos.
- **Custo e robustez.** [dual_entropico.py, linhas 18–32](../../modulos/dual_entropico.py#L18) não tem tratamento explícito de exponencial não finita e calcula Q_i @ Gamma por observável. [otimizacao.py, linhas 46–62](../../modulos/otimizacao.py#L46) repete a exponencial na saída. O passo fixo pode ser pequeno para alguns parâmetros e excessivo para outros.
- **Validação do modelo.** [script harmônico, linhas 23–46](../../tests/potencial_harmonico_sem_interacao.py#L23) valida dimensões parcialmente. Falta validação de omega real positivo, g real finito, índices distintos na faixa e pares duplicados. Índice negativo pode passar pela lista de dimensões. Em entradas válidas, a construção para (2,3) coincidiu com a fórmula em Kronecker.
- **Fachada e exemplo legado.** [__init__.py, linhas 21–22](../../modulos/__init__.py#L21) exporta dois nomes ausentes de marginais.py. Isso quebra descoberta unittest a partir da raiz e afeta importações com asterisco e o shim. [Exemplo de Ising, linhas 18 e 47](../../exemplos/dual_entropico_dois_qubits.py#L18) também referencia função ausente e usa tolerancia_residuos, que não é argumento da API atual. Imports diretos preservam o caminho novo sem exigir migração geral.
- **Ambiente.** [requeriments.txt](../../requeriments.txt) lista NumPy, SciPy, Pandas, Matplotlib e QuTiP sem versões. Na sondagem, NumPy, SciPy, Matplotlib e QuTiP estavam disponíveis. CVXPY, SDPLab e SpaceCore estavam ausentes. O benchmark SDP não foi executado.
- **Testes.** python -m unittest discover -s tests -v passou com um teste de reconstrução produto. Uma derivada direcional do dual com H interagente coincidiu com diferenças finitas a cerca de \(6{,}8\times10^{-11}\). A extensão térmica saturante foi verificada em d=2,3,5 para ambos os sinais de g. QuTiP produziu dois eixos de Wigner no teste pontual.
- **Notebooks.** exemplos.ipynb documenta primitivas e SDP. Os notebooks de casos propostos, mistura, compatibilidade e dimensão tratam de qubits e dependem da fachada. entropyc_regulaziatin.ipynb usa uma implementação independente com eigh e L-BFGS-B. tomography.ipynb trabalha com SDPLab/JAX, incluindo log-partição normalizada, uma convenção distinta da classe atual. Foram inspecionadas células e estrutura. Nenhum notebook foi executado de ponta a ponta.
