"""Problema dual para osciladores harmônicos com interações bilineares.

Usa uma base truncada de níveis de energia, com ``hbar = 1``.
"""
import matplotlib.pyplot as plt

from qutip import Qobj, wigner
import numpy as np
from collections.abc import Sequence

from modulos.dual_entropico import DualEntropico
from modulos.marginais import restricoes_lineares
from modulos.otimizacao import otimizar_dual
from modulos.operadores import embed_operator, parse_subsystem_key


def hamiltoniano_bilinear(
    dims: tuple[int, ...],
    omega: float,
    pares_interacao: Sequence[tuple[int, int, float]],
) -> np.ndarray:

    dims = tuple(dims)
    if not dims or any(not isinstance(d, (int, np.integer)) or d < 1 for d in dims):
        raise ValueError("dims deve conter dimensões inteiras positivas.")

    dimensao_global = int(np.prod(dims))
    hamiltoniano = np.zeros((dimensao_global, dimensao_global), dtype=complex)

    for indice, dimensao in enumerate(dims):
        energias = omega * (np.arange(dimensao, dtype=float) + 0.5)
        hamiltoniano += embed_operator(np.diag(energias), (indice,), dims=dims)

    pares_interacao = tuple(pares_interacao)
    subsistemas_interagentes = {
        indice for i, j, _ in pares_interacao for indice in (i, j)
    }
    posicoes = {}
    for indice in subsistemas_interagentes:
        dimensao = dims[indice]
        aniquilacao = np.diag(np.sqrt(np.arange(1, dimensao)), k=1)
        posicoes[indice] = (aniquilacao + aniquilacao.T) / np.sqrt(2.0)

    for i, j, g_ij in pares_interacao:
        hamiltoniano += embed_operator(
            g_ij * np.kron(posicoes[i], posicoes[j]), (i, j), dims=dims
        )

    return hamiltoniano


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


def visualizar_marginais(
    marginals_dict: dict[str | tuple[int, ...], np.ndarray],
    limite: float = 5.0,
    pontos: int = 101,
) -> tuple[plt.Figure, np.ndarray]:


    if not marginals_dict:
        raise ValueError("O dicionário de marginais não pode estar vazio.")

    chaves = list(marginals_dict)
    if any(len(parse_subsystem_key(chave)) != 1 for chave in chaves):
        raise ValueError("A visualização de Wigner espera marginais de um subsistema por chave.")

    x = np.linspace(-limite, limite, pontos)
    fig, axes = plt.subplots(1, len(chaves), figsize=(5 * len(chaves), 4), squeeze=False)

    for ax, chave in zip(axes[0], chaves):
        matriz = np.asarray(marginals_dict[chave], dtype=complex)
        estado = Qobj(matriz, dims=[[matriz.shape[0]], [matriz.shape[1]]])
        W = wigner(estado, x, x)
        imagem = ax.contourf(x, x, W, levels=40, cmap="RdBu_r")
        ax.set_title(f"Marginal {chave}")
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        fig.colorbar(imagem, ax=ax, label="W(x, p)")

    fig.tight_layout()
    return fig, axes[0]


if __name__ == "__main__":
    marginais = {
        "A": np.eye(2, dtype=complex) / 2,
        "B": np.eye(2, dtype=complex) / 2,
    }
    problema = construir_problema(marginais, dims=(2, 2))
    resultado = otimizar_dual(problema)
    print(f"Convergiu: {resultado['convergiu']}")
    print(f"Resíduo da normalização: {resultado['residuos'][0]:.3e}")
    print(f"Traço do estado: {np.trace(resultado['estado']).real:.12f}")
    figura, _ = visualizar_marginais(marginais)
    figura.savefig("marginais_potencial_harmonico.png", dpi=150)
    
    plt.show()
