"""Rotinas de otimização numérica para o dual entrópico."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr

from .dual_entropico import DualEntropico
from .operadores import _coordenadas_hermitianas


def _diagnostico_multiplicadores(problema: DualEntropico, tol: float = 1e-10) -> dict[str, Any]:
    """Detecta dependências lineares entre os operadores de restrição."""
    n_constraints = len(problema.Q)
    if n_constraints == 0:
        return {
            "posto_operadores": 0,
            "quantidade_multiplicadores": 0,
            "nulidade": 0,
            "indices_independentes": (),
            "indices_redundantes": (),
            "alvos_consistentes": True,
        }

    coefficients = np.asarray(
        [_coordenadas_hermitianas(operator) for operator in problema.Q],
        dtype=float,
    )
    _, triangular, pivots = qr(coefficients.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.count_nonzero(diagonal > tol * diagonal[0])) if diagonal.size else 0
    independent = tuple(sorted(int(index) for index in pivots[:rank]))
    redundant = tuple(index for index in range(n_constraints) if index not in set(independent))

    targets_consistent = True
    if independent:
        selected = coefficients[np.asarray(independent)]
        selected_targets = problema.q[np.asarray(independent)]
        for index in redundant:
            weights, *_ = np.linalg.lstsq(selected.T, coefficients[index], rcond=tol)
            implied = float(weights @ selected_targets)
            if not np.isclose(problema.q[index], implied, atol=1e-9, rtol=1e-8):
                targets_consistent = False
                break

    return {
        "posto_operadores": rank,
        "quantidade_multiplicadores": n_constraints,
        "nulidade": n_constraints - rank,
        "indices_independentes": independent,
        "indices_redundantes": redundant,
        "alvos_consistentes": targets_consistent,
    }


def otimizar_dual(
    problema: DualEntropico,
    u_inicial: Sequence[float] | None = None,
    *,
    tolerancia_residuos: float = 1e-7,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    r"""Maximiza ``D_epsilon`` pelo L-BFGS-B, minimizando ``-D_epsilon``.

    A rotina usa ``DualEntropico.avaliar`` para obter o valor e o gradiente
    analítico numa só exponencial por avaliação. O estimador inicial padrão é
    ``u=0``. Se uma tentativa intermediária exceder o intervalo numérico da
    exponencial, o otimizador recebe uma penalidade finita e uma direção de
    retorno; se a exponencial final também não for representável, a melhor
    iteração finita observada é devolvida e o resultado marca a falha.

    Convergência do L-BFGS-B e satisfação das igualdades são diagnósticos
    separados. ``restricoes_satisfeitas`` é calculado apenas após comparar os
    resíduos com ``tolerancia_residuos``; uma conclusão ``True`` é numérica,
    não uma garantia simbólica de igualdade exata.

    Multiplicadores associados a operadores linearmente dependentes não são
    únicos: combinações no núcleo da aplicação ``u -> sum_i u_i Q_i`` podem
    deixar o dual inalterado quando os alvos são compatíveis. A função relata
    posto, nulidade e consistência dos alvos, mas mantém a parametrização
    fornecida. Prefira remover redundâncias com
    :func:`construir_restricoes_marginais`.

    Parameters
    ----------
    problema:
        Instância validada de :class:`DualEntropico`.
    u_inicial:
        Estimativa inicial real dos multiplicadores; por padrão é o vetor zero.
    tolerancia_residuos:
        Tolerância absoluta máxima para classificar as restrições como satisfeitas.
    options:
        Opções adicionais do ``scipy.optimize.minimize`` para L-BFGS-B. Os
        padrões são ``maxiter=2000``, ``ftol=1e-15``, ``gtol=1e-10`` e ``maxls=40``.

    Returns
    -------
    dict
        Multiplicadores, operador global reconstruído, valor dual, resíduos,
        indicadores separados de convergência e satisfação das restrições e
        o diagnóstico do SciPy.
    """
    if not isinstance(problema, DualEntropico):
        raise TypeError("problema deve ser uma instância de DualEntropico.")
    if not np.isfinite(tolerancia_residuos) or tolerancia_residuos < 0:
        raise ValueError("tolerancia_residuos deve ser finita e não negativa.")

    if u_inicial is None:
        initial = np.zeros(len(problema.Q), dtype=float)
    else:
        initial = problema._validar_u(u_inicial).copy()

    try:
        problema.avaliar(initial)
    except (OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "A estimativa inicial produz uma exponencial instável; forneça u_inicial "
            "com operador efetivo numericamente representável."
        ) from exc

    redundancy = _diagnostico_multiplicadores(problema)
    if redundancy["nulidade"]:
        warnings.warn(
            "Há operadores de restrição linearmente redundantes; os multiplicadores "
            "podem não ser únicos e o L-BFGS-B pode apresentar direções planas.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not redundancy["alvos_consistentes"]:
        warnings.warn(
            "Os alvos associados a operadores redundantes são incompatíveis; as "
            "restrições não podem ser satisfeitas simultaneamente.",
            RuntimeWarning,
            stacklevel=2,
        )

    best: dict[str, Any] = {"u": initial.copy(), "objective": np.inf}
    overflow_penalty = 1e100
    overflow_gradient_scale = 1e50

    def objective_and_gradient(u: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            value, gradient = problema.avaliar(u)
            objective = -value
            objective_gradient = -gradient
            if not np.isfinite(objective) or not np.all(np.isfinite(objective_gradient)):
                raise OverflowError("Valor ou gradiente fora do intervalo finito.")
            if objective < best["objective"]:
                best["u"] = np.asarray(u, dtype=float).copy()
                best["objective"] = float(objective)
            return float(objective), objective_gradient
        except (OverflowError, np.linalg.LinAlgError):
            # Penalidade finita para que uma tentativa extrema não interrompa
            # o L-BFGS-B. O gradiente aponta para multiplicadores menores.
            max_component = float(np.max(np.abs(u))) if u.size else 0.0
            direction = u / max(max_component, 1.0)
            direction_norm = float(np.linalg.norm(direction))
            penalty = overflow_penalty + min(max_component, 1e50) * 1e50
            gradient = (
                overflow_gradient_scale * direction / direction_norm
                if direction_norm > 0 else np.zeros_like(u)
            )
            return float(penalty), gradient

    solver_options: dict[str, Any] = {
        "maxiter": 2000,
        "ftol": 1e-15,
        "gtol": 1e-10,
        "maxls": 40,
    }
    if options is not None:
        solver_options.update(dict(options))

    result = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options=solver_options,
    )

    final_u = np.asarray(result.x, dtype=float)
    try:
        state = problema.estado(final_u)
        selected_u = final_u
        numerical_success = bool(result.success)
    except (OverflowError, np.linalg.LinAlgError):
        selected_u = np.asarray(best["u"], dtype=float)
        state = problema.estado(selected_u)
        numerical_success = False

    expected = np.asarray(
        [np.real(np.trace(operator @ state)) for operator in problema.Q], dtype=float
    )
    residuals = expected - problema.q
    with np.errstate(over="ignore", invalid="ignore"):
        dual_value = float(
            np.dot(selected_u, problema.q) - problema.epsilon * np.trace(state).real
        )
    if not np.isfinite(dual_value) or not np.all(np.isfinite(residuals)):
        raise OverflowError("A solução final não possui valor dual/resíduos representáveis.")

    constraints_satisfied = bool(
        residuals.size == 0 or np.max(np.abs(residuals)) <= tolerancia_residuos
    )
    message = str(result.message)
    if numerical_success and not constraints_satisfied:
        message += (
            " O otimizador convergiu numericamente, mas os resíduos excedem "
            "a tolerância das restrições."
        )
    elif not numerical_success and constraints_satisfied:
        message += " Os resíduos estão dentro da tolerância, apesar do status do solver."

    return {
        "multiplicadores": selected_u.copy(),
        "estado": state,
        "valor_dual": dual_value,
        "residuos": residuals,
        "norma_residuos_inf": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        "convergencia_numerica": numerical_success,
        "restricoes_satisfeitas": constraints_satisfied,
        "tolerancia_residuos": float(tolerancia_residuos),
        "redundancia_multiplicadores": redundancy,
        "status": int(result.status),
        "mensagem": message,
        "iteracoes": int(result.nit),
        "avaliacoes": int(result.nfev),
        "metodo": "L-BFGS-B",
        "resultado_scipy": result,
    }
