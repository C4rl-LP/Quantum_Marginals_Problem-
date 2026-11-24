"""Problema dual para osciladores harmônicos com interações bilineares.

Usa uma base truncada de níveis de energia, com ``hbar = 1``.
"""
import matplotlib.pyplot as plt

import numpy as np
from collections.abc import Sequence

from modulos.dual_entropico import DualEntropico
from modulos.marginais import restricoes_lineares
from modulos.otimizacao import otimizar_dual
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.visualizacao import visualizar_marginais


def construir_problema(
    marginals_dict: dict[str | tuple[int, ...], np.ndarray],
    dims: tuple[int, ...],
    omega: float = 1.0,
    epsilon: float = 1.0,
    pares_interacao: Sequence[tuple[int, int, float]] = (),
) -> DualEntropico:

    dims = tuple(dims)
    if not dims or any(not isinstance(d, (int, np.integer)) or d < 1 for d in dims):
        raise ValueError("dims deve conter dimensões inteiras positivas.")

    hamiltoniano = hamiltoniano_bilinear(dims, omega, pares_interacao)

    observaveis, alvos = restricoes_lineares(marginals_dict, dims)

    return DualEntropico(
        H=hamiltoniano,
        Q=observaveis,
        q=alvos,
        epsilon=epsilon,
    )


if __name__ == "__main__":
    marginais = {
        "A": np.eye(2, dtype=complex) / 2,
        "B": np.eye(2, dtype=complex) / 2,
    }
    problema = construir_problema(marginais, dims=(2, 2))
    resultado = otimizar_dual(problema, tolerancia=1e-10, maxiter=5000)
    print(f"Convergiu: {resultado['convergiu']}")
    print(f"Resíduo da normalização: {resultado['residuos'][0]:.3e}")
    print(f"Traço do estado: {np.trace(resultado['estado']).real:.12f}")
    figura, _ = visualizar_marginais(marginais)
    figura.savefig("marginais_potencial_harmonico.png", dpi=150)
    
    plt.show()
