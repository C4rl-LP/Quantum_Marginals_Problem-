# 1. Diagnóstico científico

## Busca restrita por marginais de subsistemas distinguíveis

Para marginais individuais válidas, o produto é admissível. Em dimensão finita a fibra de extensões é convexa e compacta. Portanto, a minimização linear de V atinge o mínimo. A identidade do traço parcial fixa a contribuição de H₀ em toda a fibra. Ao minimizar também sobre todas as marginais individuais, a união das fibras cobre todas as densidades globais e recupera a menor energia de H. Misturar extensões ótimas demonstra a convexidade de F_V no conjunto de tuplas de marginais. Essa propriedade vale no espaço das matrizes e pode se perder numa parametrização não linear.

Usar no relatório **funcional de energia por busca restrita para marginais de subsistemas distinguíveis, análogo à RDMFT**. Aqui o dado é uma tupla de reduções de traço um de sites rotulados. Na RDMFT eletrônica o dado é uma 1-RDM coletiva, com domínio condicionado pela estatística das partículas. No projeto, V é fixo ao definir o funcional e sua expectativa exige informações de dois corpos, embora a busca restrita use apenas marginais individuais como entrada. Compatibilidade torna-se trivial neste recorte, enquanto a seleção energética de extensões continua não trivial. Para o contexto histórico, ver [Levy, busca restrita e funcionais de 1-RDM](https://pmc.ncbi.nlm.nih.gov/articles/PMC411802/) e [Gilbert, potenciais externos não locais](https://doi.org/10.1103/PhysRevB.12.2111).

## Ótimo quântico para térmicas idênticas

Escreva P=(a−a†)/(i√2). Para marginais térmicas no contínuo, as médias locais de X e P são zero, suas variâncias são ambas \(t=\bar n+1/2\), e \([X,P]=iI\). Denote \(c=\langle X_A X_B\rangle\), \(k=\langle P_A P_B\rangle\). Os pares \(X_\pm=(X_A\pm X_B)/\sqrt2,\ P_\pm=(P_A\pm P_B)/\sqrt2\) têm variâncias \(t\pm c,\ t\pm k\). Robertson implica \((t\pm c)(t\pm k)\ge 1/4\). Dividindo pelas variâncias de X e somando,

\[
2t\ge \frac14\left(\frac1{t+c}+\frac1{t-c}\right),
\qquad |c|\le \sqrt{t^2-\tfrac14}.
\]

A prova vale para estados mistos e não gaussianos com essas marginais. Os momentos cruzados são finitos por Cauchy–Schwarz e o termo de covariância simetrizada, se incluído na desigualdade de incerteza, só fortalece o limite. A série comprimida de dois modos tem marginais geométricas, \(\langle a_A a_B\rangle=\sinh r\cosh r\) e \(\langle a_A^\dagger a_B\rangle=0\). Ela satura o limite. A paridade de um modo muda o sinal de c preservando a marginal. Logo

\[
E_{\min}^{(\infty)}=2\omega(\bar n+\tfrac12)-|g|\sqrt{\bar n(\bar n+1)}.
\]

Um SDP finito pode reproduzir o ótimo no corte e serve como verificação independente. O argumento de incerteza e a construção saturante estabelecem a optimalidade também no contínuo.

### Referência quântica exata na base truncada

Para cortes iguais \(d\ge2\), \(q=\bar n/(1+\bar n)\) e \(p_j=q^j/\sum_{\ell=0}^{d-1}q^\ell\), use o operador de aniquilação truncado do script. Defina \(A=\langle a_d^\dagger a_d\rangle=\sum_j jp_j\), \(B=\langle a_da_d^\dagger\rangle=\sum_{j=0}^{d-2}(j+1)p_j\). Tem-se \(A=qB\), variâncias \(t_d=(A+B)/2\), e comutador médio \(i\kappa_d\), com \(\kappa_d=B-A=1-dp_{d-1}\). A prova anterior com \(1/4\) substituído por \(\kappa_d^2/4\) dá \(|c|\le\sqrt{AB}\). Se \(\kappa_d=0\), Cauchy–Schwarz dá \(|c|\le t_d=\sqrt{AB}\) diretamente. A extensão \(\sum_j\sqrt{p_j}\,s^j|j,j\rangle\) tem \(c=s\sum_{j=1}^{d-1}j\sqrt{p_jp_{j-1}}=s\sqrt{AB}\), com \(s=-\operatorname{sgn}(g)\) para \(g\ne0\). Assim,

\[
E_{\min}^{(d)}=2\omega(A+\tfrac12)-|g|\sqrt{AB}.
\]

Esse é o principal benchmark do solver. A referência contínua usa \(\bar n\); no corte, a energia local usa \(A\). A diferença inclui tanto o erro da base como a mudança das marginais causada pelo corte. O comutador truncado explica por que substituir apenas \(\bar n\) por A na fórmula contínua produz outra referência.

## Limite separável e relaxação PPT

Num ensemble de produtos no contínuo, em cada modo \(\operatorname{Var}_j(X)\operatorname{Var}_j(P)\ge1/4\). A variância total de P limita \(\mathbb E[\operatorname{Var}_j(P)]\le t\), e Jensen dá \(\mathbb E[\operatorname{Var}_j(X)]\ge1/(4t)\). Pela decomposição da variância, as médias condicionais de X têm variância no máximo \(t-1/(4t)\). Em cada produto a expectativa cruzada é o produto dessas médias, logo Cauchy–Schwarz no ensemble dá \(|c|\le t-1/(4t)\). O argumento vale para misturas e integrais de produtos com os momentos finitos impostos pelas marginais. Equivalentemente, para toda extensão PPT com esses momentos, a transposição reverte \(P_B\), mantém \(X_B\) e troca o sinal de k. Aplicar Robertson ao modo com variância de X igual a \(t-|c|\) antes e depois da transposição dá \((t-|c|)(t-|k|)\ge1/4\), suficiente para o mesmo limite separável.

A saturação é física. Em cada parte prepare um gaussiano puro comprimido com variâncias \((1/(4t),t)\) e médias nulas. Desloque as posições por uma variável gaussiana compartilhada de variância \(t-1/(4t)\), com sinais iguais ou opostos. A média do ensemble tem marginais gaussianas centradas com covariâncias \(tI\), ou seja, exatamente térmicas. A covariância cruzada de X iguala o limite. Portanto \(E_{\rm sep}^{(\infty)}=2\omega t-|g|(t-1/(4t))\) e \(E_{\min}\le E_{\rm sep}\le E_{\rm prod}=2\omega t\). Há vantagem estrita para \(g\ne0,\bar n>0\). A prova do ótimo separável usa tanto o limite quanto uma extensão admissível que o atinge.

No corte igual, o argumento PPT usa \(\kappa_d^2/4\) e fornece a cota \(|c|\le t_d-\kappa_d^2/(4t_d)\) para \(t_d>0\). Ela dá um limite inferior de energia para separáveis e vale no mesmo corte. Saturação finita deve ser demonstrada caso a caso. Projetar a mistura gaussiana global pode alterar as marginais condicionadas pela projeção conjunta. Para obter uma testemunha numérica, compare uma extensão admissível com o limite inferior analítico ou com **limite dual certificado** do SDP PPT. O valor de um candidato primal PPT é limite superior para o ótimo PPT. Sempre \(E_{\min}\le E_{\rm PPT}\le E_{\rm sep}\le E_{\rm prod}\). PPT equivale a separabilidade em \(2\times2\) e \(2\times3\), e em geral é somente condição necessária. Negatividade positiva certifica NPT no corte, e negatividade zero é compatível com emaranhamento PPT. [Horodecki et al., critério em dimensão finita](https://arxiv.org/abs/quant-ph/9605038). [Simon, transposição em variáveis contínuas](https://arxiv.org/abs/quant-ph/9909044).

## Regularização e interpretação dos resultados

Defina \(E_*=E_{\min}\), \(P_\varepsilon(\Gamma)=\operatorname{Tr}(H\Gamma)-\varepsilon S(\Gamma)\). Para um otimizador regularizado **admissível e exato** em dimensão D finita e \(\varepsilon>0\), comparando com um minimizador físico \(\Gamma_*\),

\[
0\le E(\Gamma_\varepsilon)-E_*
\le \varepsilon[S(\Gamma_\varepsilon)-S(\Gamma_*)]
\le\varepsilon\log D.
\]

Com uma extensão admissível \(\hat\Gamma\) e suboptimalidade regularizada certificada \(\delta\), a cota de energia vira \(\varepsilon\log D+\delta\). Na fibra fixa, subaditividade dá \(S(\Gamma)\le\sum_i S(\gamma_i)=S(\Gamma_{\rm prod})\). Isso fornece uma cota de regularização mais estreita e, por comparação do objetivo regularizado com o produto, demonstra \(E(\Gamma_\varepsilon)\le E_{\rm prod}\). Em \(g=0\) o produto é o minimizador entrópico único para \(\varepsilon>0\). Para \(\varepsilon\) crescente, a entropia e a energia físicas dos ótimos exatos são não decrescentes, por duas comparações cruzadas dos objetivos. Negatividade não tem monotonicidade análoga demonstrada aqui.

Com resíduos, a diferença \(E(\Gamma_u)-E_*\) pode ter sinal negativo e mistura erro físico e erro de restrições. No código, \(D_{\rm cod}(u)=u^\top q-\varepsilon\operatorname{Tr}\Gamma_u\), \(\Gamma_u=\exp((\sum u_\alpha Q_\alpha-H)/\varepsilon)\). Isso é o dual com regularizador \(\operatorname{Tr}(\Gamma\log\Gamma-\Gamma)\); pela imposição de traço um o valor dual na convenção do relatório é \(D_{\rm cod}+\varepsilon\). Com \(r_\alpha=\operatorname{Tr}(Q_\alpha\Gamma_u)-q_\alpha\), a identidade em aritmética exata é

\[
P_\varepsilon(\Gamma_u)-(D_{\rm cod}(u)+\varepsilon)
=u^\top r+\varepsilon(\operatorname{Tr}\Gamma_u-1).
\]

Nesse diagnóstico espectral usa-se a expressão \(\operatorname{Tr}\Gamma_u\log\Gamma_u\) mesmo se o traço ainda divergir de um. Um número pequeno ou negativo nessa identidade, sem factibilidade, não certifica optimalidade. Uma extensão admissível conhecida fornece cota superior para o primal regularizado. O dual corrigido fornece cota inferior e também \(E_*\ge D_{\rm cod}+\varepsilon\), pois a entropia dos estados admissíveis é não negativa. Para uma cota certificada mais forte, use um candidato factível reconstruído ou a solução analítica, conforme o caso.

## Modos normais e busca externa

No contínuo, \(H=(\omega/2)P^\top P+\tfrac12 X^\top(\omega I+G)X\), com \(G_{ii}=0,\ G_{ij}=g_{ij}\). O fator \(1/2\) conta uma vez cada par. Para \(\omega>0\) a estabilidade confinante exige \(\omega I+G\succ0\). Uma direção nula é um modo livre sem estado fundamental normalizável, e uma direção negativa torna a energia ilimitada inferiormente. Se \(k_j\) são os autovalores de \(\omega I+G\), as frequências são \(\sqrt{\omega k_j}\). Para dois osciladores \(k_\pm=\omega\pm g\), o que confirma a fórmula do enunciado. O corte finito continua tendo menor autovalor para parâmetros instáveis, mas isso é uma propriedade do problema finito. Para bases de Fock aninhadas, \(E_0^{(d)}\ge E_0^{(\infty)}\) e cai ao ampliar a base por Rayleigh–Ritz. A aproximação do fundamental gaussiano pelas projeções de Fock converge no regime estritamente estável.

A restrição externa a uma família oferece um limite superior de \(E_0\) quando se usa a energia física de um estado global admissível. Minimizar \(P_\varepsilon\) escolhe em geral outro estado e outro parâmetro. O mínimo regularizado não constitui cota superior para a energia fundamental. No contínuo, para a família térmica idêntica, a minimização explícita de \(2\omega t-|g|\sqrt{t^2-1/4}\) dá \(\sqrt{\omega^2-g^2/4}\) no regime estável. Marginais reais do fundamental gaussiano são térmicas comprimidas e iguais por simetria. Com \(v_x=\frac14\sum_{\pm}\sqrt{\omega/(\omega\pm g)}\), \(v_p=\frac14\sum_{\pm}\sqrt{(\omega\pm g)/\omega}\), os parâmetros são \(\bar n=\sqrt{v_xv_p}-1/2\), \(r=\tfrac14\log(v_p/v_x)\) para a convenção do enunciado. Essa família contém as marginais exatas no contínuo. No corte, a família obtida por compressão finita pode excluir as marginais do fundamental truncado.
