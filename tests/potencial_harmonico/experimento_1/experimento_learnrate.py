"""Compara taxas de aprendizado com as marginais do estado fundamental."""

import sys
from pathlib import Path

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA_EXPERIMENTO.parents[2]))

from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import estado_fundamental
from modulos.marginais import restricoes_lineares
from modulos.operadores import partial_trace
from tests.potencial_harmonico.experimento_1.Experimento_1 import dims, omega, pares_interacao
from tests.potencial_harmonico.marginais_termicas.experimento_learnrate import ler_forma, varrer_learnrate


def main(forma):
    H = hamiltoniano_bilinear(dims, omega, pares_interacao)
    _, gamma0 = estado_fundamental(H)
    marginais = {(i,): partial_trace(gamma0, keep=(i,), dims=dims) for i in range(len(dims))}
    Q, q = restricoes_lineares(marginais, dims)

    def criar_problema(epsilon, forma):
        return DualEntropico(H, Q, q, epsilon, forma=forma)

    destino = PASTA_EXPERIMENTO / f"erro_por_learnrate_{forma}.png"
    varrer_learnrate(criar_problema, forma, "Marginais do estado fundamental", destino)


if __name__ == "__main__":
    main(ler_forma())
