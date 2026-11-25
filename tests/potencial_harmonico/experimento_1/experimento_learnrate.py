"""Compara taxas de aprendizado com as marginais do estado fundamental."""

import sys
from pathlib import Path

# Permite executar o arquivo diretamente ou via python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.potencial_harmonico.experimento_1.base import PASTA, ExperimentoHarmonico
from tests.potencial_harmonico.potencial_harmonico import ler_forma, varrer_learnrate


def main(forma):
    experimento = ExperimentoHarmonico()
    destino = PASTA / f"erro_por_learnrate_{forma}.png"
    varrer_learnrate(experimento.criar_problema, forma, "Marginais do estado fundamental", destino)


if __name__ == "__main__":
    main(ler_forma(__doc__))
