from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import time
# Bibliotecas SpaceCore e SDPLab para Programação Semidefinida 
import spacecore
from spacecore import Context, DenseVectorSpace, HermitianSpace, NumpyOps
import sdplab
from sdplab import DenseConstraintOp, SDPProblem
from sdplab.solvers import run_cvxpy_solver

np.set_printoptions(precision=4, suppress=True)

# Qubits da base computacional
zero = np.array([1.0, 0.0], dtype=complex)
one  = np.array([0.0, 1.0], dtype=complex)

def ketbra(psi: np.ndarray) -> np.ndarray:
    """
    Constrói o operador projetor |psi><psi| para um vetor de estado |psi>.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return np.outer(psi, psi.conj())

def verifica_densidade(rho: np.ndarray, tol: float = 1e-10) -> Dict[str, Any]: 
    rho = np.asarray(rho, dtype=complex)
    hermitiana = bool(np.allclose(rho, rho.conj().T, atol=tol))
    autovalores = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    positiva = bool(np.all(autovalores >= -tol))
    normalizada = bool(np.isclose(np.trace(rho), 1.0, atol=tol))
    
    return {
        "hermitiana": hermitiana,
        "positiva": positiva,
        "traco_1": normalizada,
        "autovalores": autovalores
    }

def tensor(*args: np.ndarray) -> np.ndarray:
    """
    Calcula o produto tensorial sequencial (Kronecker) de múltiplos operadores ou vetores.
    """
    resultado = args[0]
    for op in args[1:]:
        resultado = np.kron(resultado, op)
    return resultado

def ghz(theta: float = 0.0) -> np.ndarray:
    estado = (
        tensor(zero, zero, zero)
        + np.exp(1j * theta) * tensor(one, one, one)
    ) / np.sqrt(2.0)
    return estado

def partial_trace(
    rho: np.ndarray,
    keep: Optional[Sequence[int]] = None,
    trace_out: Optional[Sequence[int]] = None,
    dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
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

def base_hermitiana(d: int) -> List[np.ndarray]:
    
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
    """
    Normaliza identificadores de subsistemas como 'AB', 'AC', 'BC' ou (0, 1), (0, 2).
    """
    if isinstance(key, (tuple, list)):
        return tuple(int(x) for x in key)
    if isinstance(key, str):
        char_map = {chr(ord('A') + i): i for i in range(26)}
        key_upper = key.upper().strip()
        return tuple(char_map[ch] for ch in key_upper if ch in char_map)
    raise TypeError(f"Formato de subsistema inválido: {key!r}")

def build_qmp_sdp(
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    include_trace_norm: bool = True,
    ctx: Optional[Context] = None,
) -> Tuple[SDPProblem, np.ndarray, np.ndarray]:
    r"""
    Constrói a formulação de Programação Semidefinida (SDP) para o problema de
    representabilidade de marginais quânticas no formato nativo da biblioteca SDPLab.
    """
    # 1. Normalizar as chaves dos subsistemas e matrizes marginais
    normalized_marginals: Dict[Tuple[int, ...], np.ndarray] = {}
    for k, v in marginals_dict.items():
        sub_tuple = parse_subsystem_key(k)
        mat = np.asarray(v, dtype=complex)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"A marginal {k} deve ser uma matriz quadrada; formato {mat.shape}.")
        normalized_marginals[sub_tuple] = mat

    if dims is None:
        max_idx = max(max(sub) for sub in normalized_marginals.keys())
        dims = [2] * (max_idx + 1)
    dims_tuple = tuple(dims)
    d_total = int(np.prod(dims_tuple))

    M_list: List[np.ndarray] = []
    b_list: List[float] = []

    # 2. Restrição de normalização: Tr(rho) = Tr(I_d rho) = 1.0
    if include_trace_norm:
        M_list.append(np.eye(d_total, dtype=complex))
        b_list.append(1.0)

    # 3. Restrições afins das marginais
    for sistema, omega_S in normalized_marginals.items():
        d_S = omega_S.shape[0]
        expected_d_S = int(np.prod([dims_tuple[s] for s in sistema]))
        if d_S != expected_d_S:
            raise ValueError(
                f"Dimensão da marginal para o subsistema {sistema} é {d_S}, "
                f"mas esperava-se {expected_d_S} conforme dims={dims_tuple}."
            )

        base_S = base_hermitiana(d_S)
        for E in base_S:
            val = float(np.real(np.trace(E @ omega_S)))
            M_global = embed_operator(E, sistema, dims_tuple)
            M_list.append(M_global)
            b_list.append(val)

    M_full = np.stack(M_list, axis=0)  # Formato (m, d_total, d_total)
    b_full = np.array(b_list, dtype=float)

    # 4. Configuração dos espaços no SDPLab
    if ctx is None:
        ctx = Context(NumpyOps(), dtype="complex128", check_level="none")

    dom = HermitianSpace(d_total, ctx=ctx)
    cod = DenseVectorSpace((len(b_full),), ctx=ctx)

    # Pareamento Frobenius: (A X)_i = Tr(T_i^T X) => T_i = M_i^T via swapaxes
    A = DenseConstraintOp(np.swapaxes(M_full, -1, -2), dom, cod, ctx)

    # Custo zero (C = 0) para o problema de factibilidade pura
    sdp = SDPProblem(dom.zeros(), A, b_full, ctx=ctx)

    return sdp, M_full, b_full

def solve_qmp(
    sdp: SDPProblem,
    solver: str = "CLARABEL",
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    
    try:
        X, y, prob = run_cvxpy_solver(
            sdp,
            solver=solver,
            verbose=verbose,
            return_problem=True,
            **kwargs,
        )
        status = prob.status
        is_feasible = (status in ("optimal", "optimal_inaccurate"))
        rho_mat = np.asarray(X)
        # Garante simetria hermitiana perfeita numérica
        rho_mat = (rho_mat + rho_mat.conj().T) / 2.0

        return {
            "status": status,
            "is_feasible": is_feasible,
            "rho": rho_mat,
            "dual_y": np.asarray(y),
            "problem": prob,
            "error_message": None,
        }
    except ValueError as err:
        err_str = str(err)
        status = "infeasible" if "infeasible" in err_str.lower() else "failed"
        return {
            "status": status,
            "is_feasible": False,
            "rho": None,
            "dual_y": None,
            "problem": None,
            "error_message": err_str,
        }


def verify_marginals(
    rho: Optional[np.ndarray],
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    
    if rho is None:
        return {"is_feasible": False, "message": "Nenhuma matriz rho fornecida (problema infactível)."}

    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        num_qubits = int(round(np.log2(rho.shape[0])))
        dims = [2] * num_qubits
    dims = tuple(dims)

    herm_err = float(np.linalg.norm(rho - rho.conj().T, "fro"))
    tr_val = complex(np.trace(rho))
    tr_err = float(abs(tr_val - 1.0))

    rho_sym = (rho + rho.conj().T) / 2.0
    evals = np.linalg.eigvalsh(rho_sym)
    min_eval = float(np.min(evals))
    is_psd = bool(min_eval >= -tol)
    purity = float(np.real(np.trace(rho_sym @ rho_sym)))

    marginal_errors: Dict[str, Dict[str, Any]] = {}
    for k, omega in marginals_dict.items():
        sub_tuple = parse_subsystem_key(k)
        omega_np = np.asarray(omega, dtype=complex)
        calc_marginal = partial_trace(rho_sym, keep=sub_tuple, dims=dims)
        diff = calc_marginal - omega_np
        marginal_errors[str(k)] = {
            "frobenius_error": float(np.linalg.norm(diff, "fro")),
            "max_abs_error": float(np.max(np.abs(diff))),
            "calculated_marginal": calc_marginal,
            "target_marginal": omega_np,
        }

    return {
        "is_feasible": True,
        "hermitian_error": herm_err,
        "trace": tr_val,
        "trace_error": tr_err,
        "eigenvalues": evals,
        "min_eigenvalue": min_eval,
        "is_psd": is_psd,
        "purity": purity,
        "marginal_errors": marginal_errors,
    }

def random_density_matrix(d: int) -> np.ndarray:
    """
    Gera uma matriz densidade aleatória de dimensão d usando o método de Ginibre.

    A construção é:
        G = matriz complexa d×d de entradas i.i.d. N(0,1) + i N(0,1)
        rho = G G† / Tr(G G†)

    Garantias numéricas verificadas:
        - rho = rho†   (hermiticidade)
        - rho >= 0     (positividade semidefinida)
        - Tr(rho) = 1  (normalização)

    Parameters
    ----------
    d : int
        Dimensão do espaço de Hilbert (d=2 para qubit, d=4 para 2 qubits).
    rng : np.random.Generator, opcional
        Gerador de números aleatórios (para reprodutibilidade).

    Returns
    -------
    rho : np.ndarray, shape (d, d), dtype complex128
        Matriz densidade válida.
    """
    

    # Gera matriz complexa aleatória
    G = np.random.standard_normal((d, d)) + 1j * np.random.standard_normal((d, d))

    # Constrói rho = G G†
    rho = G @ G.conj().T

    # Normaliza pelo traço
    rho = rho / np.trace(rho)

    # Garante hermiticidade perfeita numericamente (elimina erros de arredondamento)
    rho = (rho + rho.conj().T) / 2.0

    return rho  