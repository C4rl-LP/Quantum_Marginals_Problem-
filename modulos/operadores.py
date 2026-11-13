"""Operações matriciais para sistemas quânticos multipartidos."""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


def partial_trace(
    rho: np.ndarray,
    keep: Optional[Sequence[int]] = None,
    trace_out: Optional[Sequence[int]] = None,
    dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Traça os subsistemas descartados de ``rho``.

    Informe exatamente um entre ``keep`` e ``trace_out``. Os índices seguem a
    ordem dos fatores em ``dims``; sem dimensões explícitas, assume qubits.
    """
    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        num_qubits = int(round(np.log2(rho.shape[0])))
        dims = [2] * num_qubits
    dims = tuple(dims)
    N = len(dims)

    if keep is not None and trace_out is not None:
        raise ValueError("Especifique apenas `keep` ou `trace_out`, não ambos.")
    if keep is not None:
        keep_systems = tuple(sorted(keep))
    elif trace_out is not None:
        trace_set = set(trace_out)
        keep_systems = tuple(sorted(i for i in range(N) if i not in trace_set))
    else:
        raise ValueError("É necessário especificar `keep` ou `trace_out`.")

    complement = tuple(i for i in range(N) if i not in keep_systems)

    # Tensor de ordem 2N: N índices bra e N índices ket
    rho_tensor = rho.reshape(dims + dims)

    bra_indices = list(range(N))
    ket_indices = list(range(N, 2 * N))

    # Para os subsistemas traçados fora, colapsa bra e ket no mesmo índice mudo
    for c in complement:
        ket_indices[c] = bra_indices[c]

    out_bra = [bra_indices[s] for s in keep_systems]
    out_ket = [ket_indices[s] for s in keep_systems]

    # Contração tensorial direta
    reduced_tensor = np.einsum(rho_tensor, bra_indices + ket_indices, out_bra + out_ket)
    d_out = int(np.prod([dims[s] for s in keep_systems]))
    return reduced_tensor.reshape(d_out, d_out)


def _coordenadas_hermitianas(operator: np.ndarray) -> np.ndarray:
    """Retorna coordenadas de Hilbert-Schmidt na ordem de ``base_hermitiana``.

    Para uma matriz hermitiana, usa diagonais seguidas pelos componentes real e
    imaginário de cada entrada acima da diagonal, escalados por ``sqrt(2)``.
    Esta forma compacta evita materializar a base global para detectar posto.
    """
    matrix = np.asarray(operator, dtype=complex)
    diagonal = np.real(np.diag(matrix))
    off_diagonal = [
        component
        for row in range(matrix.shape[0])
        for column in range(row + 1, matrix.shape[1])
        for component in (
            np.sqrt(2.0) * np.real(matrix[row, column]),
            np.sqrt(2.0) * np.imag(matrix[row, column]),
        )
    ]
    return np.concatenate((diagonal, np.asarray(off_diagonal, dtype=float)))


def base_hermitiana(d: int) -> List[np.ndarray]:
    """Gera uma base ortonormal, no produto de Hilbert-Schmidt, de Herm(d)."""
    
    bases: List[np.ndarray] = []
    # Elementos diagonais
    for i in range(d):
        E_ii = np.zeros((d, d), dtype=complex)
        E_ii[i, i] = 1.0
        bases.append(E_ii)

    # Elementos fora da diagonal
    for i in range(d):
        for j in range(i + 1, d):
            # Parte real
            E_real = np.zeros((d, d), dtype=complex)
            E_real[i, j] = 1.0 / np.sqrt(2.0)
            E_real[j, i] = 1.0 / np.sqrt(2.0)
            bases.append(E_real)

            # Parte imaginária
            E_imag = np.zeros((d, d), dtype=complex)
            E_imag[i, j] = 1.0j / np.sqrt(2.0)
            E_imag[j, i] = -1.0j / np.sqrt(2.0)
            bases.append(E_imag)

    return bases


def embed_operator(
    O_s: np.ndarray,
    sistema: Sequence[int],
    dim: Optional[Sequence[int]] = None,
    *,
    dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""
    Incorpora um operador local O_s atuando no subsistema S (`sistema`) no espaço
    global H = prod_k H_k, inserindo identidades nos subsistemas do complemento S^c.
    """
    if dim is None:
        dim = dims
    if dim is None:
        raise ValueError("É necessário especificar as dimensões locais `dim` (ou `dims`).")
    dims_tuple = tuple(dim)
    Nt = len(dims_tuple)
    sistema_tuple = tuple(sistema)
    Ns = len(sistema_tuple)
    complemento = tuple(i for i in range(Nt) if i not in sistema_tuple)

    dimensao_s = tuple(dims_tuple[i] for i in sistema_tuple)
    O_tensor = O_s.reshape(dimensao_s + dimensao_s)

    # Produto externo com identidade para cada sistema complementar
    res = O_tensor
    for c in complemento:
        I_c = np.eye(dims_tuple[c], dtype=O_s.dtype)
        res = np.multiply.outer(res, I_c)

    # Mapeamento dos eixos originais para a ordem global
    axis_map_bra = {s: idx for idx, s in enumerate(sistema_tuple)}
    axis_map_ket = {s: Ns + idx for idx, s in enumerate(sistema_tuple)}
    for idx, c in enumerate(complemento):
        axis_map_bra[c] = 2 * Ns + 2 * idx
        axis_map_ket[c] = 2 * Ns + 2 * idx + 1

    perm = [axis_map_bra[i] for i in range(Nt)] + [axis_map_ket[i] for i in range(Nt)]
    res = np.transpose(res, perm)
    d_total = int(np.prod(dims_tuple))
    return res.reshape(d_total, d_total)


def parse_subsystem_key(key: Union[str, Sequence[int]], num_qubits: int = 3) -> Tuple[int, ...]:
    """Converte letras (A, B, ...) ou índices em uma tupla de subsistemas.

    Exemplos: ``'AB'`` vira ``(0, 1)`` e ``'C'`` vira ``(2,)``.
    """
    if isinstance(key, (tuple, list)):
        return tuple(int(x) for x in key)
    if isinstance(key, str):
        char_map = {chr(ord('A') + i): i for i in range(26)}
        key_upper = key.upper().strip()
        return tuple(char_map[ch] for ch in key_upper if ch in char_map)
    raise TypeError(f"Formato de subsistema inválido: {key!r}")
