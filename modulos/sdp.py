"""Formulação e resolução do SDP de extensão de marginais quânticas."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from spacecore import Context, DenseVectorSpace, HermitianSpace, NumpyOps
from sdplab import DenseConstraintOp, SDPProblem
from sdplab.solvers import run_cvxpy_solver
from .operadores import base_hermitiana, embed_operator, parse_subsystem_key


def build_qmp_sdp(
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    include_trace_norm: bool = True,
    ctx: Optional[Context] = None,
) -> Tuple[SDPProblem, np.ndarray, np.ndarray]:
    r"""Monta o problema SDP de factibilidade para as marginais informadas.

    Retorna o objeto SDPLab e também os observáveis globais ``M`` e alvos ``b``
    usados para impor ``Tr(M_i rho) = b_i`` no formato nativo do SDPLab.
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
    """Executa o SDP e retorna status, estado primal, variáveis duais e erro."""
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
