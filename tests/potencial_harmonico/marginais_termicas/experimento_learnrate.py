"""Compara o resíduo final do dual para diferentes passos de gradiente."""

import sys
from pathlib import Path

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA_EXPERIMENTO.parents[2]))

from modulos.estados import estado_termico_oscilador
from tests.potencial_harmonico.potencial_harmonico import construir_problema, ler_forma, varrer_learnrate


def main(forma="exponencial"):
    dims = (4, 4)
    omega = 1.0
    pares_interacao = [(0, 1, 0.5)]

    def criar_problema(epsilon, forma):
        marginais = {
            (i,): estado_termico_oscilador(dim, temperatura=epsilon, omega=omega)
            for i, dim in enumerate(dims)
        }
        return construir_problema(marginais, dims, omega, epsilon, pares_interacao, forma=forma)

    nome = "erro_por_learnrate_log.png" if forma == "logaritmico" else "erro_por_learnrate_2.png"
    varrer_learnrate(criar_problema, forma, r"$T_{\mathrm{local}} = \epsilon$", PASTA_EXPERIMENTO / nome)


if __name__ == "__main__":
    main(ler_forma(__doc__))
