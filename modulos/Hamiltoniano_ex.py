from collections.abc import Sequence

import numpy as np

from .operadores import embed_operator, operador_posicao
from .validacoes import validar_dimensoes


PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.diag([1, -1])


def hamiltoniano_oscilador(dimensao, omega=1.0):
    """Retorna o Hamiltoniano local diagonal na base de Fock truncada."""
    (dimensao,) = validar_dimensoes((dimensao,))
    return np.diag(omega * (np.arange(dimensao, dtype=float) + 0.5))


def hamiltoniano_bilinear(
    dims: tuple[int, ...],
    omega: float,
    pares_interacao: Sequence[tuple[int, int, float]],
) -> np.ndarray:
    """Soma as energias locais e as interações na convenção -g_ij X_i X_j."""

    dims = tuple(dims)
    if not dims or any(not isinstance(d, (int, np.integer)) or d < 1 for d in dims):
        raise ValueError("dims deve conter dimensões inteiras positivas.")

    dimensao_global = int(np.prod(dims))
    hamiltoniano = np.zeros((dimensao_global, dimensao_global), dtype=complex)

    for indice, dimensao in enumerate(dims):
        local = hamiltoniano_oscilador(dimensao, omega)
        hamiltoniano += embed_operator(local, (indice,), dims=dims)

    pares_interacao = tuple(pares_interacao)
    subsistemas_interagentes = {
        indice for i, j, _ in pares_interacao for indice in (i, j)
    }
    posicoes = {indice: operador_posicao(dims[indice]) for indice in subsistemas_interagentes}

    for i, j, g_ij in pares_interacao:
        hamiltoniano -= embed_operator(
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
