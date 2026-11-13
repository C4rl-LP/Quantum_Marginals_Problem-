"""Estados de referência e operações elementares em vetores e operadores."""

import numpy as np


zero = np.array([1.0, 0.0], dtype=complex)
one = np.array([0.0, 1.0], dtype=complex)

def basis(d, k):
    """Retorna o vetor computacional |k> em dimensão ``d`` (índice iniciado em zero)."""
    e = np.zeros(d, dtype=complex)
    e[k] = 1.0
    return e


def ketbra(psi: np.ndarray) -> np.ndarray:
    """Converte um vetor de estado em seu projetor |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return np.outer(psi, psi.conj())


def tensor(*args: np.ndarray) -> np.ndarray:
    """Calcula o produto de Kronecker na mesma ordem dos argumentos."""
    resultado = args[0]
    for op in args[1:]:
        resultado = np.kron(resultado, op)
    return resultado


def ghz(theta: float = 0.0) -> np.ndarray:
    """Constrói o vetor GHZ de três qubits com fase relativa ``theta``."""
    estado = (
        tensor(zero, zero, zero)
        + np.exp(1j * theta) * tensor(one, one, one)
    ) / np.sqrt(2.0)
    return estado


def mestura(rho_1, rho_2, t, dim=None):
    """Retorna ``t*rho_1 + (1-t)*rho_2`` sem validar o intervalo de ``t``.

    ``dim`` é mantido apenas para compatibilidade com chamadas antigas e não é
    necessário para a combinação matricial.
    """
    return t * rho_1 + (1 - t) * rho_2
