# Quantum Marginal Problem (QMP) via SDP

Este projeto investiga o **Problema das Marginais Quânticas (Quantum Marginal Problem - QMP)** para sistemas multipartidos de qubits. O objetivo central é determinar computacionalmente se um conjunto de matrizes densidade reduzidas (marginais locais) é compatível com a existência de um estado quântico global $\rho$.

Enquanto a representabilidade de estados puros ($\operatorname{rank}(\rho)=1$) é um problema não convexo, a **representabilidade de estados mistos** ($\rho \succeq 0, \operatorname{Tr}\rho = 1$) pode ser formulada e resolvida de forma exata e eficiente como um problema de **Programação Semidefinida (SDP)**.

---

## 🔬 Formulação Matemática

Para um sistema de três qubits $A, B, C$, a matriz densidade global $\rho_{ABC} \in \mathbb{C}^{8 \times 8}$ deve satisfazer:

1. **Positividade (LMI):** $\rho_{ABC} \succeq 0$ (matriz semidefinida positiva);
2. **Normalização:** $\operatorname{Tr}(\rho_{ABC}) = 1$;
3. **Consistência das Marginais:** $\operatorname{Tr}_{S^c}(\rho_{ABC}) = \omega_S$ para cada subsistema $S$.

Como o traço parcial é uma transformação linear nos elementos de $\rho$, o problema de representabilidade mista é um problema de factibilidade semidefinida convexo:

$$\begin{aligned}
\text{encontrar} \quad & \rho_{ABC} \in \operatorname{Herm}(8) \\
\text{sujeito a} \quad & \rho_{ABC} \succeq 0, \\
& \operatorname{Tr}(\rho_{ABC}) = 1, \\
& \operatorname{Tr}_{S^c}(\rho_{ABC}) = \omega_S.
\end{aligned}$$

---

## 🧪 Exemplos Implementados

A implementação em [`qmp_sdplab.py`](qmp_sdplab.py) avalia os três regimes fundamentais do projeto:

1. **Estado GHZ:**
   - Marginais: $\omega_{AB} = \omega_{AC} = \omega_{BC} = \frac{1}{2}(|00\rangle\langle00| + |11\rangle\langle11|)$.
   - **Resultado:** **Factível** (`optimal`). O SDP encontra a mistura clássica $\rho_{\mathrm{cl}} = \frac{1}{2}(|000\rangle\langle000| + |111\rangle\langle111|)$, que possui exatamente as mesmas marginais de 2 corpos que o estado puro $|\mathrm{GHZ}_\theta\rangle$.
   - *Regime:* Mixed = Sim, Pure = Sim.

2. **Marginais Maximamente Mistas:**
   - Marginais: $\omega_{AB} = \omega_{AC} = \omega_{BC} = \frac{I_4}{4}$.
   - **Resultado:** **Factível** (`optimal`). O SDP recupera o estado global maximamente misto $\rho_{ABC} = \frac{I_8}{8}$.
   - *Regime:* Mixed = Sim, Pure = Não (nenhum estado puro tripartite pode ter marginais de 2 qubits maximamente mistas).

3. **Incompatibilidade (Monogamia do Entrelaçamento):**
   - Marginais: $\omega_{AB} = |\Phi^+\rangle\langle\Phi^+|$ e $\omega_{AC} = |\Phi^+\rangle\langle\Phi^+|$ com $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$.
   - **Resultado:** **Infactível** (`infeasible`). O solver cônico fornece um certificado dual de infactibilidade. Fisicamente, um qubit não pode estar simultaneamente em um estado puro maximamente entrelaçado com dois sistemas distintos (monogamia do entrelaçamento / forte subaditividade).
   - *Regime:* Mixed = Não, Pure = Não.

---

## 🚀 Como Executar

### Pré-requisitos

Recomenda-se o uso do ambiente virtual com as dependências instaladas:

```bash
# Ativar o ambiente virtual
source .venv/bin/activate

# Instalar dependências principais (garantindo compatibilidade de spacecore com sdplab)
pip install "spacecore==0.4.2" sdplab cvxpy clarabel scs numpy scipy
```

### Executando a Simulação

Para rodar os três exemplos com o solver padrão de pontos interiores (**CLARABEL**):

```bash
python qmp_sdplab.py
```

Ou alternativamente especificando outro solver cônico (como o **SCS**):

```bash
python qmp_sdplab.py SCS
```

---

## 📁 Estrutura do Repositório

* [`qmp_sdplab.py`](qmp_sdplab.py): Módulo principal reutilizável contendo as funções:
  - `partial_trace(...)`: cálculo geral de traço parcial via contração tensorial (`np.einsum`);
  - `build_qmp_sdp(...)`: montagem do problema cônico no formato nativo da biblioteca `sdplab`;
  - `solve_qmp(...)`: execução do solver cônico (`run_cvxpy_solver`) e tratamento de factibilidade/infactibilidade;
  - `verify_marginals(...)`: auditoria numérica completa (positividade, traço, autovalores, pureza e erro das marginais).
* [`Rascunho.pdf`](Rascunho.pdf): Documento teórico preliminar sobre a formulação do QMP e mapeamento matricial para SDP.
* [`inicial_teste.ipynb`](inicial_teste.ipynb): Notebook de exploração preliminar.