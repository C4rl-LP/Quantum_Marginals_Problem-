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


def estado_fundamental(H: np.ndarray) -> tuple[float, np.ndarray]:
    """Retorna a menor energia e o projetor de um autovetor fundamental."""
    energias, vetores = np.linalg.eigh(H)
    return float(energias[0]), ketbra(vetores[:, 0])


def estado_gibbs(H: np.ndarray, beta: float) -> np.ndarray:
    """Retorna o estado térmico normalizado à temperatura inversa beta >= 0."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta deve ser finito e não negativo.")
    energias, vetores = np.linalg.eigh(H)
    pesos = np.exp(-beta * (energias - energias[0]))
    pesos /= pesos.sum()
    return (vetores * pesos) @ vetores.conj().T


def estado_termico_oscilador(dimensao, temperatura=1.0, omega=1.0):
    """Retorna o Gibbs local do oscilador em unidades com k_B = 1."""
    from .Hamiltoniano_ex import hamiltoniano_oscilador
    from .validacao import validar_escalar_positivo

    temperatura = validar_escalar_positivo(temperatura, "temperatura local")
    return estado_gibbs(hamiltoniano_oscilador(dimensao, omega), beta=1 / temperatura)


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

def entropia_von_neumann(estado):
    """Calcula a entropia de von Neumann em nats."""
    autovalores = np.linalg.eigvalsh(estado)
    positivos = autovalores[autovalores > 0]
    return float(-np.sum(positivos * np.log(positivos)))


def negatividade(estado):
    """Retorna a negatividade na bipartição do primeiro qubit e o restante."""
    estado = np.asarray(estado)
    if estado.ndim != 2 or estado.shape[0] != estado.shape[1]:
        raise ValueError("estado deve ser uma matriz quadrada.")
    tamanho = estado.shape[0]
    if tamanho < 4 or tamanho % 2 != 0:
        raise ValueError("estado deve ter dimensão compatível com a bipartição 2 × d, d >= 2.")
    resto = tamanho // 2
    matriz = estado.reshape(2, resto, 2, resto)
    transposta_parcial = matriz.transpose(0, 3, 2, 1).reshape(tamanho, tamanho)
    autovalores = np.linalg.eigvalsh(transposta_parcial)
    return float(np.abs(autovalores[autovalores < 0]).sum())
