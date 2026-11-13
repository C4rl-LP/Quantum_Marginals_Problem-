"""Diagnósticos numéricos de estados e de extensões marginais."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from .operadores import parse_subsystem_key, partial_trace


def verifica_densidade(rho: np.ndarray, tol: float = 1e-10) -> Dict[str, Any]: 
    """Verifica hermiticidade, positividade e traço unitário dentro de ``tol``."""
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


def _verify_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> Dict[str, Any]:
    """Compatibilidade com o formato antigo de diagnóstico booleano/erro.

    A validação é delegada a :func:`verifica_densidade`, evitando manter duas
    implementações independentes dos mesmos critérios físicos.
    """
    resultado = verifica_densidade(rho, tol=tol)
    autovalores = resultado["autovalores"]
    hermitian_error = float(np.linalg.norm(np.asarray(rho) - np.asarray(rho).conj().T, "fro"))
    trace_error = float(abs(np.trace(np.asarray(rho)) - 1.0))
    min_eigenvalue = float(np.min(autovalores))
    return {
        "hermitian_error": hermitian_error,
        "trace_error": trace_error,
        "min_eigenvalue": min_eigenvalue,
        "is_valid": resultado["hermitiana"] and resultado["positiva"] and resultado["traco_1"],
    }



def verify_marginals(
    rho: Optional[np.ndarray],
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    """Compara reduções de ``rho`` com as marginais-alvo e retorna diagnósticos.

    O campo ``is_feasible`` informa se uma matriz foi fornecida; os erros e
    propriedades físicas são reportados separadamente no restante do resultado.
    """
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
