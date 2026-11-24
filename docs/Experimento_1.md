# Experimento 1 — correlações entre dois qubits

## Objetivo

Comparar estados globais de dois qubits que têm as mesmas marginais locais, \(\rho_A=\rho_B=I_2/2\), mas são submetidos a Hamiltonianos diferentes. O experimento ilustra como a interação pode criar correlações e, no caso XY, emaranhamento.

## Método

O estado global é obtido numericamente pela otimização dual entrópica, impondo traço um e as marginais locais fixadas. Comparam-se três Hamiltonianos: sem interação, Ising \(H_{ZZ}=-J\,\sigma_z\otimes\sigma_z\) e XY \(H_{XY}=-J(\sigma_x\otimes\sigma_x+\sigma_y\otimes\sigma_y)\), com \(J>0\).

Para cada caso, o programa compara a matriz reconstruída com a previsão analítica e informa o erro de Frobenius, a norma dos resíduos das restrições, a pureza e a negatividade. Os parâmetros padrão são \(J=1\) e \(\varepsilon=0.5\); ambos podem ser alterados no início de `main` em `tests/Experimento_1.py`.

## Previsões

- **Sem interação:** a solução de máxima entropia é \(I_4/4\), sem correlações.
- **Ising ZZ:** aparecem correlações clássicas, com \(\Gamma=(I_4+\tanh(J/\varepsilon)\,\sigma_z\otimes\sigma_z)/4\). Esse estado é separável.
- **XY:** o estado é o Gibbs normalizado \(e^{-H_{XY}/\varepsilon}/\operatorname{Tr}(e^{-H_{XY}/\varepsilon})\). Quando \(\varepsilon\to0^+\), ele tende ao estado emaranhado \(|\Psi^+\rangle=(|01\rangle+|10\rangle)/\sqrt{2}\).

## Interpretação

As marginais não determinam sozinhas o estado global. A interação seleciona as correlações, enquanto \(\varepsilon\) controla o peso da entropia em relação à energia. A comparação numérica deve ser interpretada junto dos resíduos e da convergência do otimizador: um estado só é uma reconstrução confiável quando também satisfaz as restrições com erro pequeno.
