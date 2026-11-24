from collections.abc import Sequence

import numpy as np

from modulos.operadores import embed_operator


PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.diag([1, -1])



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

def dimensoes_qubits(N):
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N deve ser um inteiro maior ou igual a 2.")
    return (2,) * N


def hamiltoniano_vizinhos(J, N, termo_local):
    """Soma o termo de interação sobre os N-1 pares de uma cadeia aberta."""
    dims = dimensoes_qubits(N)
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H -= J * embed_operator(termo_local, (i, i + 1), dims=dims)
    return H


def Hamiltoniano_ZZ(J, N=2):
    return hamiltoniano_vizinhos(J, N, np.kron(PAULI_Z, PAULI_Z))


def Hamiltoniano_XY(J, N=2):
    termo = np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y)
    return hamiltoniano_vizinhos(J, N, termo)
