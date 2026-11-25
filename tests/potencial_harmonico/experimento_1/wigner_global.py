"""Compara cortes da Wigner global do estado fundamental e da reconstrução."""

import argparse

import numpy as np

from gerar_resultados_slides import DESTINO, salvar

import matplotlib.pyplot as plt
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import estado_fundamental
from modulos.marginais import restricoes_lineares
from modulos.operadores import partial_trace
from modulos.otimizacao import otimizar_dual
from modulos.visualizacao import wigner_marginal
from tests.potencial_harmonico.experimento_1.Experimento_1 import dims, omega, pares_interacao


def cortes_wigner(estado, grade):
    # A Wigner conjunta retorna os eixos (p_A, x_A, p_B, x_B).
    posicoes = wigner_marginal(estado, grade, [0.0], keep=(0, 1), dims=dims)
    momentos = wigner_marginal(estado, [0.0], grade, keep=(0, 1), dims=dims)
    return posicoes[0, :, 0, :].T, momentos[:, 0, :, 0].T


def main(epsilon):
    H = hamiltoniano_bilinear(dims, omega, pares_interacao)
    _, referencia = estado_fundamental(H)
    marginais = {(i,): partial_trace(referencia, keep=(i,), dims=dims) for i in range(len(dims))}
    Q, q = restricoes_lineares(marginais, dims)
    problema = DualEntropico(H, Q, q, epsilon, forma="logaritmico")

    tolerancia = 1e-6
    resultado = otimizar_dual(
        problema, metodo="gradient", learnrate=0.1 * epsilon,
        tolerancia=tolerancia, maxiter=10000,
    )
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
        eixos, cortes_wigner(referencia, grade), cortes_wigner(estado, grade), rotulos,
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

    DESTINO.mkdir(exist_ok=True)
    salvar(figura, f"wigner_global_epsilon_{epsilon:g}")
    print(f"ε={epsilon:g} | resíduo: {residuo:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=100)
    main(parser.parse_args().epsilon)
