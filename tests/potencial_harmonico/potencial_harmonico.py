"""Problema dual para osciladores harmônicos com interações bilineares.

Usa uma base truncada de níveis de energia, com ``hbar = 1``.
"""
import numpy as np
from collections.abc import Sequence

from modulos.dual_entropico import DualEntropico
from modulos.marginais import restricoes_lineares
from modulos.estados import estado_fundamental, estado_gibbs
from modulos.operadores import momentos_observaveis_locais, operador_posicao
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.validacoes import validar_dimensoes, validar_escalar_positivo, validar_marginais


def construir_problema(
    marginals_dict: dict[str | tuple[int, ...], np.ndarray],
    dims: tuple[int, ...],
    omega: float = 1.0,
    epsilon: float = 1.0,
    pares_interacao: Sequence[tuple[int, int, float]] = (),
    *,
    forma: str = "exponencial",
) -> DualEntropico:

    dims = tuple(dims)
    if not dims or any(not isinstance(d, (int, np.integer)) or d < 1 for d in dims):
        raise ValueError("dims deve conter dimensões inteiras positivas.")

    hamiltoniano = hamiltoniano_bilinear(dims, omega, pares_interacao)

    observaveis, alvos = restricoes_lineares(marginals_dict, dims)

    return DualEntropico(
        H=hamiltoniano,
        Q=observaveis,
        q=alvos,
        epsilon=epsilon,
        forma=forma,
    )


def marginal_termica(dimensao, epsilon=1, omega=1):
    """Prepara o estado térmico local, com epsilon como temperatura em k_B = 1."""
    (dimensao,) = validar_dimensoes((dimensao,))
    temperatura = validar_escalar_positivo(epsilon, "temperatura local")
    H_local = np.diag(omega * (np.arange(dimensao) + 0.5))
    return estado_gibbs(H_local, beta=1 / temperatura)


def _momentos_posicao(estado_completo, marginais, dims):
    dims = validar_dimensoes(dims)
    marginais = validar_marginais(marginais, dims)
    if len(marginais) != 2:
        raise ValueError("A correlação de posição exige dois modos distintos.")
    if any(len(sistema) != 1 for sistema in marginais):
        raise ValueError("Cada marginal deve identificar um único modo.")

    sistemas = tuple(i for (i,) in marginais)
    posicoes = tuple(operador_posicao(dims[i]) for i in sistemas)
    return momentos_observaveis_locais(estado_completo, dims, sistemas, posicoes)


def correlacao_posicao(estado_completo, marginais, dims):
    """Calcula a covariância de posição entre dois modos distintos."""
    return _momentos_posicao(estado_completo, marginais, dims)["covariancia"]


def pearson_posicao(estado_completo, marginais, dims):
    """Calcula Pearson usando as variâncias locais do próprio estado."""
    pearson = _momentos_posicao(estado_completo, marginais, dims)["pearson"]
    if np.isnan(pearson):
        raise ValueError("A correlação de Pearson exige variâncias de posição positivas.")
    return pearson
