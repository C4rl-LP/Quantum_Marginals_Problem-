# 7. Plano experimental

Fixar inicialmente \(N=2\), \(\omega=1\) como unidade de energia, cortes iguais, e \(g/\omega=0{,}4\), \(\bar n=0{,}3\), \(\varepsilon/\omega=0{,}1\), \(d=4\) como ponto piloto proposto. Usar \(g=0\), \(-g\) e vácuo como controles. Fazer uma varredura por vez. A direção inicial de \(|g|/\omega\) vai de zero até 0,8, ainda no regime estável. Usar \(d=3,4,6,8\) para triagem, e expandir conforme cauda, diagnóstico e custo. Triar \(\bar n\) entre 0 e 1 e epsilon num conjunto geométrico decrescente a partir de 0,2 até 0,01 em unidades de omega, apenas enquanto os critérios numéricos forem atendidos. Testar compressão e amplitude coerente após o piloto. Para coerentes, começar com p=1/2 e alpha real. Explorar N=3 com d pequeno só após a validação bipartida.

| Gráfico | Controle e referência | Leitura esperada e limite da inferência |
|---|---|---|
| Energia condicionada × g | Fixar d, \(\bar n,\omega,\varepsilon\). Produto e ótimo finito | Produto plano para médias nulas, ótimo não regularizado linear em \(-|g|\). A energia regularizada mede o estado selecionado |
| Ganho × \(\bar n\) | Fixar g, d e epsilon. Comparar \(E_{\rm prod}-E_{\min}^{(d)}\) | Ganho entre zero e a diferença exata para ótimo regularizado factível. Ganho sozinho pode ser clássico |
| Produto, PPT/separável e quântico × \(\bar n\) | Primeiro mostrar as três referências contínuas. Separadamente, marcar produto, ótimo, limite analítico e benchmark PPT no corte | A janela entre os limites quantifica vantagem quântica quando a comparação usa a mesma fibra. Não interpretar cota PPT como ótimo separável geral |
| Negatividade × g | Fixar d, \(\bar n,\varepsilon\), usar produto, extensão saturante e limiar energético como controles | Zero no produto e em g=0. Surgimento de NPT nos estados regularizados é hipótese a testar. Usar margem acima do erro espectral |
| \(F_V\) × \(\bar n\) | Fixar g e d. Comparar com a expressão finita. Plotar energia de interação reconstruída e valor regularizado de interação com rótulos distintos | Separa contribuição de V da energia local que cresce com ocupação. Curva parametrizada não implica convexidade em \(\bar n\) |
| Superfície de energia em \((\bar n,r)\) | Fixar g, d, epsilon. Marcar mínimo da grade e parâmetros contínuos obtidos dos modos normais | Compressão pode diminuir a energia variacional. A cobertura do contínuo é demonstrada para o fundamental gaussiano; a melhora no corte deve ser medida |
| Energia fundamental × g | Comparar diagonalização, modos normais e energia física dos candidatos validados da busca externa | Identificar erro de família, corte e regularização mediante comparações na mesma dimensão antes do limite contínuo |
| Erro de regularização × epsilon | Fixar H, d e alvos. Plotar diferença ao ótimo finito, cota \(\varepsilon\log D\), erro marginal e traço | Para solução exata, energia física tende ao ótimo e fica dentro da cota. Estagnação com resíduo pode refletir o solver |
| Erro de truncagem × d | Fixar \(\bar n\) do contínuo e g. Usar diferença entre ótimos condicionados finito e contínuo e \(q^d\). Usar outro painel para o fundamental truncado | Monotonicidade variacional vale para a energia fundamental em bases aninhadas. O problema condicionado muda de alvos ao variar d |
| Tempo × D | Registrar N,d,m, backend, hardware e contagem de avaliações. Separar tempo de montar, resolver e diagnosticar | Identificar gargalo observado. Faixa finita de medidas não determina complexidade assintótica |

## Visualização de correlações

Reutilizar visualizar_marginais para inspecionar térmicas, misturas e compressão. Para um alvo fixo, produto e extensão admissível têm as mesmas funções de Wigner locais. Comparar essas funções visualmente serve como checagem de reconstrução, com controlo quantitativo pelo traço parcial.

Priorizar a matriz de covariância de \((X_A,P_A,X_B,P_B)\) e sua diferença para o produto. Usar \(\sigma_{ij}=\langle\{R_i-\langle R_i\rangle,R_j-\langle R_j\rangle\}/2\rangle\). Monitorar as correlações de posição e momento, usando os operadores truncados. Um gráfico de posição conjunta pode ser calculado no espaço contínuo usando funções de Hermite da base de Fock, ou em autovalores discretos dos operadores X truncados. Os dois gráficos representam medições diferentes e devem trazer rótulo e normalização próprios. Para o contínuo, integrar numericamente e comparar as reduções com as densidades de posição locais e com uma referência de quadratura.

Deixar cortes da Wigner conjunta para depois. Um corte numa seção de espaço de fase não é uma distribuição marginal. A Wigner não negativa é compatível com emaranhamento gaussiano. Use negatividade da transposta parcial, cotas separáveis e análise de covariância com hipóteses declaradas.
