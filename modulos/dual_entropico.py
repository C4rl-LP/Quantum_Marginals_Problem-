"""Dual do problema de energia com regularização entrópica.

Para ``Tr(Gamma log(Gamma) - Gamma)``, a conjugada entrópica produz
``Gamma_u = exp((sum_i u_i Q_i - H) / epsilon)`` sem deslocamento por ``-I``.
Os resíduos são definidos como valores esperados reconstruídos menos prescritos;
por isso, são o oposto do gradiente da função dual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_HERMITIAN_ATOL = 1e-10
_HERMITIAN_RTOL = 1e-10


def _as_finite_matrix(value: object, name: str) -> np.ndarray:
    """Converte uma entrada em matriz complexa quadrada, finita e não vazia."""
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} deve ser uma matriz numérica quadrada.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} deve ser uma matriz quadrada não vazia; recebido {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} deve conter apenas valores finitos.")
    if not np.allclose(matrix, matrix.conj().T, atol=_HERMITIAN_ATOL, rtol=_HERMITIAN_RTOL):
        raise ValueError(f"{name} deve ser hermitiana.")

    # Remove apenas assimetrias dentro da tolerância numérica aceita acima.
    result = matrix / 2 + matrix.conj().T / 2
    result.setflags(write=False)
    return result


def _as_real_vector(value: object, name: str) -> np.ndarray:
    """Converte uma entrada em vetor real e finito, sem aceitar partes imaginárias."""
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} deve ser uma sequência numérica unidimensional.") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} deve ser unidimensional; recebido shape {vector.shape}.")
    if np.iscomplexobj(vector) and np.any(np.imag(vector) != 0):
        raise ValueError(f"{name} deve conter valores reais.")
    try:
        result = np.asarray(np.real(vector), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} deve conter valores numéricos reais.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} deve conter apenas valores finitos.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DualEntropico:
    r"""Avalia o dual entrópico associado a restrições de expectativa.

    O funcional primal usa exatamente
    ``Tr(H Gamma) + epsilon Tr(Gamma log(Gamma) - Gamma)``. Para operadores
    Hermitianos ``H`` e ``Q_i``, o estado não normalizado associado a ``u`` é

    ``Gamma_u = exp((sum_i u_i Q_i - H) / epsilon)``.

    A dimensão de cada operador é a dimensão global do espaço de Hilbert. A
    classe não normaliza ``Gamma_u``: o primal informado também não impõe
    ``Tr(Gamma) = 1`` a menos que isso esteja incluído entre as restrições.

    Parameters
    ----------
    H:
        Hamiltoniano hermitiano global, de dimensão ``d x d``.
    Q:
        Sequência de operadores hermitianos globais ``Q_i``, cada um ``d x d``.
    q:
        Valores esperados prescritos, vetor real de comprimento ``len(Q)``.
    epsilon:
        Parâmetro de regularização estritamente positivo e finito.
    """

    H: np.ndarray
    Q: Sequence[np.ndarray]
    q: Sequence[float]
    epsilon: float

    def __post_init__(self) -> None:
        hamiltonian = _as_finite_matrix(self.H, "H")

        try:
            operators_input = tuple(self.Q)
        except TypeError as exc:
            raise ValueError("Q deve ser uma sequência de operadores.") from exc
        operators = tuple(_as_finite_matrix(operator, f"Q[{index}]")
                          for index, operator in enumerate(operators_input))
        for index, operator in enumerate(operators):
            if operator.shape != hamiltonian.shape:
                raise ValueError(
                    f"Q[{index}] tem dimensão {operator.shape}; esperava-se {hamiltonian.shape}."
                )

        prescribed = _as_real_vector(self.q, "q")
        if prescribed.size != len(operators):
            raise ValueError(
                f"q tem comprimento {prescribed.size}, mas Q contém {len(operators)} operadores."
            )

        try:
            epsilon_array = np.asarray(self.epsilon)
            if epsilon_array.ndim != 0 or np.iscomplexobj(epsilon_array):
                raise ValueError
            epsilon = float(epsilon_array)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("epsilon deve ser um escalar real finito e positivo.") from exc
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon deve ser um escalar real finito e positivo.")

        object.__setattr__(self, "H", hamiltonian)
        object.__setattr__(self, "Q", operators)
        object.__setattr__(self, "q", prescribed)
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def dimensao(self) -> int:
        """Dimensão ``d`` dos operadores globais ``d x d``."""
        return self.H.shape[0]

    def _validar_u(self, u: Sequence[float]) -> np.ndarray:
        """Valida e converte os multiplicadores duais para um vetor real."""
        multipliers = _as_real_vector(u, "u")
        if multipliers.size != len(self.Q):
            raise ValueError(
                f"u tem comprimento {multipliers.size}; esperava-se {len(self.Q)}."
            )
        return multipliers

    def _estado_e_expectativas(self, u: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Calcula uma exponencial hermitiana e todos os valores esperados."""
        multipliers = self._validar_u(u)
        effective = -self.H.copy()
        for coefficient, operator in zip(multipliers, self.Q):
            with np.errstate(over="ignore", invalid="ignore"):
                effective += coefficient * operator
        if not np.all(np.isfinite(effective)):
            raise OverflowError("O operador efetivo excede o intervalo representável em float64.")
        effective = effective / 2 + effective.conj().T / 2

        eigenvalues, eigenvectors = np.linalg.eigh(effective)
        scaled_eigenvalues = eigenvalues / self.epsilon
        if np.any(scaled_eigenvalues > np.log(np.finfo(np.float64).max)):
            raise OverflowError(
                "A exponencial de Gamma_u excede o intervalo representável em float64."
            )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            weights = np.exp(scaled_eigenvalues)
            gamma = (eigenvectors * weights) @ eigenvectors.conj().T
        if not np.all(np.isfinite(gamma)):
            raise OverflowError("Gamma_u não é representável com valores finitos em float64.")
        with np.errstate(over="ignore", invalid="ignore"):
            gamma = gamma / 2 + gamma.conj().T / 2
        if not np.all(np.isfinite(gamma)):
            raise OverflowError("Gamma_u não é representável com valores finitos em float64.")

        expectations = np.array(
            [float(np.real(np.trace(operator @ gamma))) for operator in self.Q],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(expectations)):
            raise OverflowError("Os valores esperados de Gamma_u excedem float64.")
        return gamma, expectations

    def _calcular_valor(self, u: np.ndarray, gamma: np.ndarray) -> float:
        """Calcula o objetivo e detecta overflow no produto escalar ou no traço."""
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.dot(u, self.q) - self.epsilon * np.trace(gamma).real
        if not np.isfinite(result):
            raise OverflowError("O valor do dual excede o intervalo representável em float64.")
        return float(result)

    def estado(self, u: Sequence[float]) -> np.ndarray:
        r"""Retorna ``Gamma_u = exp((sum_i u_i Q_i - H)/epsilon)`` sem normalizar."""
        gamma, _ = self._estado_e_expectativas(u)
        return gamma

    def residuos(self, u: Sequence[float]) -> np.ndarray:
        """Retorna ``Tr(Q_i Gamma_u) - q_i`` para cada restrição."""
        _, expectations = self._estado_e_expectativas(u)
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = expectations - self.q
        if not np.all(np.isfinite(residuals)):
            raise OverflowError("Os resíduos excedem o intervalo representável em float64.")
        return residuals

    def valor(self, u: Sequence[float]) -> float:
        r"""Retorna ``u·q - epsilon Tr(Gamma_u)`` para a convenção entrópica dada."""
        multipliers = self._validar_u(u)
        gamma, _ = self._estado_e_expectativas(multipliers)
        return self._calcular_valor(multipliers, gamma)

    def gradiente(self, u: Sequence[float]) -> np.ndarray:
        """Retorna o gradiente analítico ``q_i - Tr(Q_i Gamma_u)``."""
        _, expectations = self._estado_e_expectativas(u)
        with np.errstate(over="ignore", invalid="ignore"):
            gradient = self.q - expectations
        if not np.all(np.isfinite(gradient)):
            raise OverflowError("O gradiente excede o intervalo representável em float64.")
        return gradient

    def avaliar(self, u: Sequence[float]) -> tuple[float, np.ndarray]:
        """Retorna ``(valor, gradiente)`` calculando ``Gamma_u`` uma única vez."""
        multipliers = self._validar_u(u)
        gamma, expectations = self._estado_e_expectativas(multipliers)
        value = self._calcular_valor(multipliers, gamma)
        with np.errstate(over="ignore", invalid="ignore"):
            gradient = self.q - expectations
        if not np.all(np.isfinite(gradient)):
            raise OverflowError("O gradiente excede o intervalo representável em float64.")
        return value, gradient
