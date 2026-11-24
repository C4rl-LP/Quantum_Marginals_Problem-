"""Visualizações de estados quânticos reduzidos."""

import matplotlib.pyplot as plt
import numpy as np

from .operadores import parse_subsystem_key


def visualizar_marginais(
    marginals_dict: dict[str | tuple[int, ...], np.ndarray],
    limite: float = 5.0,
    pontos: int = 101,
) -> tuple[plt.Figure, np.ndarray]:
    """Plota a função de Wigner das marginais de um único modo."""
    if not marginals_dict:
        raise ValueError("O dicionário de marginais não pode estar vazio.")

    chaves = list(marginals_dict)
    if any(len(parse_subsystem_key(chave)) != 1 for chave in chaves):
        raise ValueError("A visualização de Wigner espera marginais de um subsistema por chave.")

    from qutip import Qobj, wigner

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
