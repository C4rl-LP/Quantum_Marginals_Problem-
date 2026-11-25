"""Reconstrói o estado fundamental de dois osciladores a partir das marginais."""

import sys
from pathlib import Path

import numpy as np

# Parâmetros do experimento: altere estes valores para modificar o exemplo.
dims = (4, 4)
omega = 1.0
g = 0.5
epsilons = np.geomspace(1.0, 0.01, 21)  # inclui epsilon = 0.1
pares_interacao = [(0, 1, g)]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import entropia_von_neumann
from modulos.marginais import restricoes_lineares
from modulos.operadores import partial_trace
from modulos.otimizacao import otimizar_dual
from modulos.visualizacao import visualizar_marginais
from tests.potencial_harmonico.potencial_harmonico import correlacao_posicao


def estado_fundamental(H):
    """Retorna a menor energia e a matriz densidade do estado fundamental."""
    energias, estados = np.linalg.eigh(H)
    psi0 = estados[:, 0]
    return energias[0], np.outer(psi0, psi0.conj())


if __name__ == "__main__":
    H = hamiltoniano_bilinear(dims, omega, pares_interacao)
    energia_fundamental, gamma0 = estado_fundamental(H)

    marginais = {
        (i,): partial_trace(gamma0, keep=(i,), dims=dims)
        for i in range(len(dims))
    }
    Q, q = restricoes_lineares(marginais, dims)
    correlacao_fundamental = correlacao_posicao(gamma0, marginais, dims)
    u_inicial = None
    for epsilon in epsilons:
        problema = DualEntropico(H, Q, q, epsilon)
        resultado = otimizar_dual(
            problema,
            u_inicial=u_inicial,
            maxiter=10000,
            learnrate=0.1 * epsilon,
        )
        u_inicial = resultado["multiplicadores"]
        gamma = resultado["estado"]
        energia_reconstruida = np.trace(H @ gamma).real
        correlacao_reconstruida = correlacao_posicao(gamma, marginais, dims)

        print(
            f"epsilon={epsilon:.3e} | "
            f"E0={energia_fundamental:.10f} | "
            f"E={energia_reconstruida:.10f} | "
            f"|E-E0|={abs(energia_reconstruida - energia_fundamental):.3e} | "
            f"S={entropia_von_neumann(gamma):.8f} | "
            f"||Gamma-Gamma0||_F={np.linalg.norm(gamma - gamma0, ord='fro'):.3e} | "
            f"||resíduos||={resultado['norma_residuos']:.3e} | "
            f"convergiu={resultado['convergiu']} | "
            f"<X0 X1>={correlacao_reconstruida:.8f} "
            f"(fundamental: {correlacao_fundamental:.8f})"
        )

    marginais_reconstruidas = {
        sistema: partial_trace(gamma, keep=sistema, dims=dims)
        for sistema in marginais
    }
    figura, _ = visualizar_marginais(marginais)
    figura.suptitle("Marginais prescritas do estado fundamental")
    figura, _ = visualizar_marginais(marginais_reconstruidas)
    figura.suptitle("Marginais do estado reconstruído")

    import matplotlib.pyplot as plt

    plt.show()
