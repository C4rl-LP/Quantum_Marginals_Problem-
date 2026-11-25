"""Modelo físico e reconstrução compartilhados pelos scripts do Experimento 1."""

from functools import partial
from pathlib import Path

from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import estado_fundamental
from modulos.marginais import reduzir_marginais, restricoes_lineares
from modulos.operadores import momentos_observaveis_locais, operador_posicao
from modulos.otimizacao import varrer_dual

PASTA = Path(__file__).resolve().parent
DIMENSOES = (4, 4)
OMEGA = 1.0
ACOPLAMENTO = 0.5
MODOS = (0, 1)
METODO = "gradient"
TOLERANCIA = 1e-6
MAXITER = 10000
FATOR_LEARNRATE = 0.1


class ExperimentoHarmonico:
    """Prepara a referência e as restrições para um valor de acoplamento."""

    def __init__(self, g=ACOPLAMENTO):
        self.pares_interacao = [(*MODOS, g)]
        self.H = hamiltoniano_bilinear(DIMENSOES, OMEGA, self.pares_interacao)
        self.energia, self.estado = estado_fundamental(self.H)
        self.marginais = self.calcular_marginais(self.estado)
        self.Q, self.q = restricoes_lineares(self.marginais, DIMENSOES)
        self.posicoes = tuple(operador_posicao(DIMENSOES[i]) for i in MODOS)

    def calcular_marginais(self, estado):
        """Retorna um dicionário de matrizes densidade indexado por (modo,)."""
        return reduzir_marginais(estado, ((i,) for i in MODOS), DIMENSOES)

    def momentos_posicao(self, estado):
        """Retorna os momentos de posição, incluindo covariância e Pearson."""
        return momentos_observaveis_locais(estado, DIMENSOES, MODOS, self.posicoes)

    def criar_problema(self, epsilon, forma):
        """Retorna um DualEntropico com as marginais de referência fixas."""
        return DualEntropico(self.H, self.Q, self.q, epsilon, forma=forma)

    def reconstruir(self, epsilon, forma, u_inicial=None):
        """Retorna o dicionário do otimizador, incluindo a matriz de estado."""
        return next(self.varrer((epsilon,), forma, u_inicial=u_inicial))

    def varrer(self, epsilons, forma, **inicializacao):
        """Gera os resultados numéricos para os ε fornecidos."""
        return varrer_dual(
            partial(self.criar_problema, forma=forma), epsilons,
            fator_learnrate=FATOR_LEARNRATE, metodo=METODO,
            tolerancia=TOLERANCIA, maxiter=MAXITER, **inicializacao,
        )
