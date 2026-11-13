"""Exemplo do dual entrópico para marginais maximamente mistas de dois qubits.

Executado a partir da raiz do repositório, imprime uma comparação com a solução
analítica e salva ``imagens/dual_entropico_correlacao.png``.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Permite executar diretamente com ``python exemplos/dual_entropico_dois_qubits.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modulos.dual_entropico import DualEntropico
from modulos.marginais import construir_restricoes_marginais
from modulos.operadores import partial_trace
from modulos.otimizacao import otimizar_dual


def main() -> None:
    identity = np.eye(2, dtype=complex)
    sigma_z = np.diag([1.0, -1.0])
    zz = np.kron(sigma_z, sigma_z)
    J = 0.8

    maximally_mixed = identity / 2
    restrictions = construir_restricoes_marginais(
        dims=[2, 2],
        marginais=[maximally_mixed, maximally_mixed],
        subsistemas=[[0], [1]],
    )

    epsilon_values = np.geomspace(0.2, 5.0, 32)
    numeric_correlations = []
    exact_correlations = np.tanh(J / epsilon_values)

    for epsilon in epsilon_values:
        problem = DualEntropico(
            H=-J * zz,
            Q=restrictions.Q,
            q=restrictions.q,
            epsilon=float(epsilon),
        )
        result = otimizar_dual(problem, tolerancia_residuos=2e-8)
        if not result["restricoes_satisfeitas"]:
            raise RuntimeError(
                f"Restrições não satisfeitas para epsilon={epsilon:g}: "
                f"{result['norma_residuos_inf']:.3e}"
            )
        correlation = float(np.real(np.trace(zz @ result["estado"])))
        numeric_correlations.append(correlation)

    # Relata um ponto representativo, incluindo erro das duas marginais.
    epsilon_report = 1.0
    report_problem = DualEntropico(
        H=-J * zz,
        Q=restrictions.Q,
        q=restrictions.q,
        epsilon=epsilon_report,
    )
    report = otimizar_dual(report_problem, tolerancia_residuos=2e-8)
    exact_state = (np.eye(4) + np.tanh(J / epsilon_report) * zz) / 4
    rho_a = partial_trace(report["estado"], keep=[0], dims=[2, 2])
    rho_b = partial_trace(report["estado"], keep=[1], dims=[2, 2])
    marginal_error = max(
        np.linalg.norm(rho_a - maximally_mixed, ord="fro"),
        np.linalg.norm(rho_b - maximally_mixed, ord="fro"),
    )

    print(f"Convergência numérica: {report['convergencia_numerica']}")
    print(f"Restrições satisfeitas: {report['restricoes_satisfeitas']}")
    print(f"Valor dual (epsilon={epsilon_report:g}): {report['valor_dual']:.12g}")
    print(f"Resíduo máximo: {report['norma_residuos_inf']:.3e}")
    print(f"Erro de reconstrução analítica (Frobenius): "
          f"{np.linalg.norm(report['estado'] - exact_state, ord='fro'):.3e}")
    print(f"Erro máximo das marginais: {marginal_error:.3e}")
    print("Estado global reconstruído:")
    print(np.array2string(report["estado"], precision=8, suppress_small=True))

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogx(epsilon_values, numeric_correlations, "o", ms=4,
                  label="L-BFGS-B")
    axis.semilogx(epsilon_values, exact_correlations, "-",
                  label=r"Analítico: $\tanh(J/\varepsilon)$")
    axis.set_xlabel(r"Regularização $\varepsilon$")
    axis.set_ylabel(r"Correlação $\langle\sigma_z\otimes\sigma_z\rangle$")
    axis.set_title("Correlação térmica de dois qubits")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path = PROJECT_ROOT / "imagens" / "dual_entropico_correlacao.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    main()
