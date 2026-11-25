"""Compara o resíduo final do dual para diferentes passos de gradiente."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA_EXPERIMENTO.parents[2]))

from modulos.otimizacao import otimizar_dual
from tests.potencial_harmonico.potencial_harmonico import construir_problema, marginal_termica


def varrer_learnrate(criar_problema, forma, descricao, destino):
    epsilons = np.geomspace(.01, .1, 5)
    fatores = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    tolerancia = 1e-6
    maxiter = 10000
    residuos = np.empty((len(fatores), len(epsilons)))

    for j, epsilon in enumerate(epsilons):
        problema = criar_problema(epsilon, forma)
        for i, fator in enumerate(fatores):
            resultado = otimizar_dual(
                problema, metodo="gradient", learnrate=fator * epsilon,
                tolerancia=tolerancia, maxiter=maxiter, salvar_matriz_densidade=False,
            )
            residuos[i, j] = resultado["norma_residuos"]

    figura, eixo = plt.subplots(figsize=(9, 5), layout="constrained")
    for fator, erros in zip(fatores, residuos):
        eixo.loglog(epsilons, erros, marker="o", markersize=4, label=f"{fator:g}")
        atingiu = np.count_nonzero(erros < tolerancia)
        print(f"c={fator:g} | tolerância atingida em {atingiu}/{len(epsilons)} | ε={epsilons[0]:g}: {erros[0]:.3e}")
    eixo.axhline(tolerancia, color="0.4", linestyle="--", label="Tolerância")
    eixo.set_xlabel(r"Regularização $\epsilon$")
    eixo.set_ylabel(r"Norma do resíduo final $\|r\|_2$")
    eixo.set_title(f"Dual {forma} · {descricao}")
    eixo.grid(alpha=0.25, which="both")
    eixo.legend(title=r"Taxa $c\epsilon$", ncols=2)
    figura.savefig(destino, dpi=150)
    plt.close(figura)


def ler_forma():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forma", choices=("exponencial", "log", "logaritmico"), default="exponencial")
    args = parser.parse_args()
    return "logaritmico" if args.forma == "log" else args.forma


def main(forma="exponencial"):
    dims = (4, 4)
    omega = 1.0
    pares_interacao = [(0, 1, 0.5)]

    def criar_problema(epsilon, forma):
        marginais = {
            (i,): marginal_termica(dim, epsilon=epsilon, omega=omega)
            for i, dim in enumerate(dims)
        }
        return construir_problema(marginais, dims, omega, epsilon, pares_interacao, forma=forma)

    nome = "erro_por_learnrate_log.png" if forma == "logaritmico" else "erro_por_learnrate_2.png"
    varrer_learnrate(criar_problema, forma, r"$T_{\mathrm{local}} = \epsilon$", PASTA_EXPERIMENTO / nome)


if __name__ == "__main__":
    main(ler_forma())
