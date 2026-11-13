"""Converte marginais prescritas em observáveis globais e valores-alvo."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import Iterator, Sequence

import numpy as np
from scipy.linalg import qr

from .operadores import _coordenadas_hermitianas, base_hermitiana, embed_operator
from .validacao import verifica_densidade


@dataclass(frozen=True)
class RestricoesMarginais:
    """Restrições globais independentes e metadados da redução de redundâncias.

    ``Q`` e ``q`` são as entradas diretamente aceitas por :class:`DualEntropico`.
    Os índices de metadados referem-se à lista de restrições antes da redução.
    A iteração retorna somente ``(Q, q)`` para permitir desempacotamento simples.
    """

    Q: tuple[np.ndarray, ...]
    q: np.ndarray
    quantidade_original: int
    indices_mantidos: tuple[int, ...]
    indices_redundantes: tuple[int, ...]

    def __iter__(self) -> Iterator[object]:
        yield self.Q
        yield self.q


def construir_restricoes_marginais(
    dims: Sequence[int],
    marginais: Sequence[np.ndarray],
    subsistemas: Sequence[Sequence[int]],
    *,
    rank_tol: float = 1e-10,
    consistency_atol: float = 1e-9,
    consistency_rtol: float = 1e-8,
) -> RestricoesMarginais:
    r"""Constrói ``(Q_i, q_i)`` globais a partir de marginais prescritas.

    Para cada marginal ``rho_S``, expande os observáveis locais ``E_a`` na
    base Hermitiana ortonormal de ``Herm(d_S)`` e usa a dualidade do traço:

    ``Tr(E_a rho_S) = Tr(embed_operator(E_a, S, dims) Gamma)``.

    O embedding respeita a ordem do produto tensorial do projeto. Os índices de
    cada subsistema devem estar em ordem crescente, que é também a ordem dos
    fatores na matriz marginal. Marginais podem ser disjuntas ou sobrepostas.
    Restrições globais linearmente dependentes são removidas quando seus alvos
    obedecem a mesma combinação linear; alvos incompatíveis geram ``ValueError``.

    Parameters
    ----------
    dims:
        Dimensões locais globais, na ordem dos fatores de ``np.kron``.
    marginais:
        Matrizes densidade locais, uma para cada entrada de ``subsistemas``.
    subsistemas:
        Índices globais que identificam os fatores de cada matriz marginal.
    rank_tol:
        Tolerância relativa para determinar dependência linear dos observáveis.
    consistency_atol, consistency_rtol:
        Tolerâncias para verificar os alvos de restrições redundantes.

    Returns
    -------
    RestricoesMarginais
        Contém ``Q``, ``q`` e os índices mantidos/removidos. Pode ser
        desempacotado como ``Q, q = construir_restricoes_marginais(...)``.
    """
    try:
        dims_tuple = tuple(dims)
    except TypeError as exc:
        raise ValueError("dims deve ser uma sequência de inteiros positivos.") from exc
    if not dims_tuple or any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, Integral)
        or dimension <= 0
        for dimension in dims_tuple
    ):
        raise ValueError("dims deve conter inteiros positivos.")
    dims_tuple = tuple(int(dimension) for dimension in dims_tuple)

    try:
        marginais_tuple = tuple(marginais)
        subsistemas_tuple = tuple(tuple(indices) for indices in subsistemas)
    except TypeError as exc:
        raise ValueError("marginais e subsistemas devem ser sequências.") from exc
    if not marginais_tuple:
        raise ValueError("Informe ao menos uma marginal prescrita.")
    if len(marginais_tuple) != len(subsistemas_tuple):
        raise ValueError("Deve haver exatamente um conjunto de índices por marginal.")

    if not np.isfinite(rank_tol) or rank_tol <= 0:
        raise ValueError("rank_tol deve ser finito e positivo.")
    if not np.isfinite(consistency_atol) or consistency_atol < 0:
        raise ValueError("consistency_atol deve ser finito e não negativo.")
    if not np.isfinite(consistency_rtol) or consistency_rtol < 0:
        raise ValueError("consistency_rtol deve ser finito e não negativo.")

    full_dimension = prod(dims_tuple)
    operators: list[np.ndarray] = []
    targets: list[float] = []

    for marginal_index, (marginal_input, subsystem) in enumerate(
        zip(marginais_tuple, subsistemas_tuple)
    ):
        if not subsystem:
            raise ValueError(f"subsistemas[{marginal_index}] não pode ser vazio.")
        if any(
            isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            for index in subsystem
        ):
            raise ValueError(f"subsistemas[{marginal_index}] deve conter índices inteiros.")
        subsystem = tuple(int(index) for index in subsystem)
        if tuple(sorted(set(subsystem))) != subsystem:
            raise ValueError(
                f"subsistemas[{marginal_index}] deve ter índices distintos em ordem crescente."
            )
        if subsystem[0] < 0 or subsystem[-1] >= len(dims_tuple):
            raise ValueError(f"subsistemas[{marginal_index}] contém índice fora de dims.")

        local_dimension = int(np.prod([dims_tuple[index] for index in subsystem]))
        try:
            marginal = np.asarray(marginal_input, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"marginais[{marginal_index}] deve ser uma matriz numérica.") from exc
        if marginal.shape != (local_dimension, local_dimension):
            raise ValueError(
                f"marginais[{marginal_index}] tem shape {marginal.shape}; esperava-se "
                f"({local_dimension}, {local_dimension}) para subsistemas {subsystem}."
            )
        if not np.all(np.isfinite(marginal)):
            raise ValueError(f"marginais[{marginal_index}] deve conter valores finitos.")
        density_check = verifica_densidade(marginal)
        if not (density_check["hermitiana"] and density_check["positiva"] and density_check["traco_1"]):
            raise ValueError(
                f"marginais[{marginal_index}] deve ser uma matriz densidade "
                "(hermitiana, positiva semidefinida e de traço um)."
            )

        for local_observable in base_hermitiana(local_dimension):
            operators.append(embed_operator(local_observable, subsystem, dims=dims_tuple))
            target = np.trace(local_observable @ marginal)
            targets.append(float(np.real(target)))

    target_vector = np.asarray(targets, dtype=np.float64)
    # Coordenadas compactas equivalentes à base Hermitiana global evitam
    # materializar D^2 matrizes globais de dimensão D x D.
    coefficient_rows = np.asarray(
        [_coordenadas_hermitianas(operator) for operator in operators],
        dtype=np.float64,
    )

    # QR com pivotamento escolhe primeiro as linhas numericamente independentes.
    _, triangular, pivots = qr(coefficient_rows.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    threshold = rank_tol * (diagonal[0] if diagonal.size else 0.0)
    rank = int(np.count_nonzero(diagonal > threshold))
    kept = tuple(sorted(int(index) for index in pivots[:rank]))
    removed = tuple(index for index in range(len(operators)) if index not in set(kept))

    if kept:
        kept_rows = coefficient_rows[np.asarray(kept)]
        kept_targets = target_vector[np.asarray(kept)]
        for index in removed:
            coefficients, *_ = np.linalg.lstsq(kept_rows.T, coefficient_rows[index], rcond=rank_tol)
            operator_error = np.linalg.norm(kept_rows.T @ coefficients - coefficient_rows[index])
            if operator_error > rank_tol * max(1.0, np.linalg.norm(coefficient_rows[index])):
                raise RuntimeError("A seleção QR não reproduziu uma restrição dependente.")
            implied_target = float(np.dot(coefficients, kept_targets))
            if not np.isclose(
                target_vector[index], implied_target,
                atol=consistency_atol, rtol=consistency_rtol,
            ):
                raise ValueError(
                    "As marginais prescritas são incompatíveis: uma restrição global "
                    f"redundante (índice {index}) exige alvo {implied_target:.12g}, "
                    f"mas recebeu {target_vector[index]:.12g}."
                )

    independent_operators = tuple(operators[index] for index in kept)
    independent_targets = target_vector[np.asarray(kept, dtype=int)]
    independent_targets.setflags(write=False)
    return RestricoesMarginais(
        Q=independent_operators,
        q=independent_targets,
        quantidade_original=len(operators),
        indices_mantidos=kept,
        indices_redundantes=removed,
    )
