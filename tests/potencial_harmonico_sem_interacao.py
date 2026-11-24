"""Problema dual para osciladores harmônicos com interações bilineares.

Usa uma base truncada de níveis de energia, com ``hbar = 1``.
"""
import matplotlib.pyplot as plt

import numpy as np
from collections.abc import Sequence

from modulos.dual_entropico import DualEntropico
from modulos.marginais import restricoes_lineares
from modulos.operadores import partial_trace, embed_operator
from modulos.otimizacao import otimizar_dual
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.validacoes import validar_dimensoes, validar_marginais
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


# Vamos definir marginal termica para OH 
def marginal_termica(dimensao, epsilon=1, omega= 1):
    # Por questão numerica vamos utilizar um o log-sum-exp trick
    a = 0
    diagonais = np.exp(-(omega*(np.arange(dimensao)- a))/epsilon)
    diagonais /= np.sum(diagonais)
    # Conseguimos garantir que existe pelo menos um autovalor 1 quando epsilon vai para 0 e que este é o estado puro.
    return np.diag(diagonais)

def correlacao_posicao(estado_completo, marginais, dims):
    """Calcula a expectativa de X_i X_j para dois modos distintos."""
    dims = validar_dimensoes(dims)
    marginais = validar_marginais(marginais, dims)
    if len(marginais) != 2:
        raise ValueError("A correlação de posição exige dois modos distintos.")
    if any(len(sistema) != 1 for sistema in marginais):
        raise ValueError("Cada marginal deve identificar um único modo.")

    posicoes = []
    for sistema, marginal in marginais.items():
        a = np.diag(np.sqrt(np.arange(1, marginal.shape[0])), k=1)
        X = (a + a.T) / np.sqrt(2)
        posicoes.append(embed_operator(X, sistema, dims=dims))

    X_a, X_b = posicoes
    return float(np.einsum("ij,ji->", X_a @ X_b, estado_completo).real)

if __name__ == "__main__":
    dims = (4, 4)
    marginais = {
        (indice,): marginal_termica(dimensao)
        for indice, dimensao in enumerate(dims)
    }
    problema = construir_problema(marginais, dims=dims)
    resultado = otimizar_dual(problema, tolerancia=1e-10, maxiter=5000, learnrate=0.1)
    print(f"Convergiu: {resultado['convergiu']}")
    print(f"Resíduo da normalização: {resultado['residuos'][0]:.3e}")
    estado = resultado["estado"]
    print(f"Traço do estado: {np.trace(estado).real:.12f}")

    reconstruidas = {
        chave: partial_trace(estado, keep=chave, dims=dims)
        for chave in marginais
    }

    figura, _ = visualizar_marginais(marginais)
    figura.suptitle("Marginais prescritas")
    figura.savefig("marginais_potencial_harmonico.png", dpi=150)

    figura_reconstruida, _ = visualizar_marginais(reconstruidas)
    figura_reconstruida.suptitle("Marginais reconstruídas")
    figura_reconstruida.savefig("marginais_reconstruidas_potencial_harmonico.png", dpi=150)

    plt.show()
