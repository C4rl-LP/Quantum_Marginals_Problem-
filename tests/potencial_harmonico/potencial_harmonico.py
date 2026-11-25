"""Problema dual para osciladores harmônicos com interações bilineares.

Usa uma base truncada de níveis de energia, com ``hbar = 1``.
"""
import argparse
from collections.abc import Sequence
from functools import partial

import numpy as np

from modulos.dual_entropico import DualEntropico
from modulos.marginais import restricoes_lineares
from modulos.estados import estado_termico_oscilador
from modulos.operadores import momentos_observaveis_locais, operador_posicao
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.otimizacao import varrer_taxas_dual
from modulos.validacoes import validar_dimensoes, validar_marginais
from modulos.visualizacao import salvar_figura, visualizar_taxas_dual


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
    return estado_termico_oscilador(dimensao, temperatura=epsilon, omega=omega)


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


def varrer_learnrate(criar_problema, forma, descricao, destino):
    """Executa o protocolo harmônico de taxas e grava a figura compartilhada."""
    epsilons = np.geomspace(0.01, 0.1, 5)
    fatores = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    tolerancia = 1e-6
    residuos = varrer_taxas_dual(
        partial(criar_problema, forma=forma), epsilons, fatores,
        tolerancia=tolerancia, maxiter=10000,
    )
    for fator, erros in zip(fatores, residuos):
        atingiu = np.count_nonzero(erros < tolerancia)
        print(f"c={fator:g} | tolerância atingida em {atingiu}/{len(epsilons)} "
              f"| ε={epsilons[0]:g}: {erros[0]:.3e}")
    figura, _ = visualizar_taxas_dual(
        epsilons, fatores, residuos, tolerancia=tolerancia, titulo=f"Dual {forma} · {descricao}",
    )
    salvar_figura(figura, destino, dpi=150)


def ler_forma(descricao):
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument("--forma", choices=("exponencial", "log", "logaritmico"), default="exponencial")
    args = parser.parse_args()
    return "logaritmico" if args.forma == "log" else args.forma
