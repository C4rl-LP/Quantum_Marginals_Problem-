"""Compara cortes da Wigner global do estado fundamental e da reconstrução."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Permite executar o arquivo diretamente ou via python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modulos.visualizacao import cortes_wigner
from tests.potencial_harmonico.experimento_1.base import DIMENSOES, MODOS, ExperimentoHarmonico
from tests.potencial_harmonico.experimento_1.figuras import salvar


def main(epsilon):
    experimento = ExperimentoHarmonico()
    referencia = experimento.estado
    resultado = experimento.reconstruir(epsilon, "logaritmico")
    residuo = resultado["norma_residuos"]

    estado = resultado["estado"]
    grade = np.linspace(-3, 3, 101)
    figura, eixos = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    status = "tolerância atingida" if resultado["convergiu"] else "acima da tolerância"
    figura.suptitle(rf"Cortes da Wigner global · $\varepsilon={epsilon:g}$" "\n"
                    f"Resíduo {residuo:.2e} · {status}")
    rotulos = ((r"$x_A$", r"$x_B$", r"$p_A=p_B=0$"),
               (r"$p_A$", r"$p_B$", r"$x_A=x_B=0$"))
    for linha, alvo, reconstruido, (xlabel, ylabel, corte) in zip(
        eixos,
        cortes_wigner(referencia, grade, keep=MODOS, dims=DIMENSOES),
        cortes_wigner(estado, grade, keep=MODOS, dims=DIMENSOES),
        rotulos,
    ):
        diferenca = reconstruido - alvo
        limite = max(np.max(np.abs(alvo)), np.max(np.abs(reconstruido)))
        limite_diferenca = np.max(np.abs(diferenca))
        for eixo, dados, escala, titulo in zip(
            linha, (alvo, reconstruido, diferenca), (limite, limite, limite_diferenca),
            ("Estado fundamental", "Estado reconstruído", "Reconstruído − fundamental"),
        ):
            imagem = eixo.pcolormesh(grade, grade, dados, cmap="RdBu_r", vmin=-escala, vmax=escala)
            eixo.set(xlabel=xlabel, ylabel=ylabel, title=f"{titulo}\n{corte}", aspect="equal")
        figura.colorbar(linha[0].collections[0], ax=linha[:2], shrink=0.8, label="Wigner")
        figura.colorbar(imagem, ax=linha[2], shrink=0.8, label="Diferença")
        print(f"{corte} | diferença máxima: {limite_diferenca:.3e}")

    salvar(figura, f"wigner_global_epsilon_{epsilon:g}")
    print(f"ε={epsilon:g} | resíduo: {residuo:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=100)
    args = parser.parse_args()
    plt.switch_backend("Agg")
    main(args.epsilon)
