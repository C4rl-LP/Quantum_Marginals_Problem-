"""Compara correlações de posição para diferentes acoplamentos e regularizações."""

import numpy as np

from gerar_resultados_slides import DESTINO, salvar

import matplotlib.pyplot as plt
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import estado_fundamental
from modulos.marginais import restricoes_lineares
from modulos.operadores import momentos_observaveis_locais, operador_posicao, partial_trace
from modulos.otimizacao import otimizar_dual
from tests.potencial_harmonico.experimento_1.Experimento_1 import dims, omega


def main():
    acoplamentos = (0, 0.25, 0.5, 0.75, 0.9)
    epsilons = np.geomspace(0.01, 1, 21)
    modos = (0, 1)
    posicoes = tuple(operador_posicao(dims[i]) for i in modos)
    medidas = ("covariancia", "pearson")
    tolerancia = 1e-6
    figura, eixos = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True, layout="constrained")

    for g in acoplamentos:
        H = hamiltoniano_bilinear(dims, omega, [(0, 1, g)])
        _, fundamental = estado_fundamental(H)
        marginais = {(i,): partial_trace(fundamental, keep=(i,), dims=dims) for i in modos}
        Q, q = restricoes_lineares(marginais, dims)
        referencia = momentos_observaveis_locais(fundamental, dims, modos, posicoes)
        dados = []
        residuos = []
        for epsilon in epsilons:
            problema = DualEntropico(H, Q, q, epsilon, forma="logaritmico")

            resultado = otimizar_dual(
                problema, metodo="gradient", learnrate=0.1 * epsilon,
                tolerancia=tolerancia, maxiter=10000,
            )
            residuos.append(resultado["norma_residuos"])
            estado = resultado["estado"]
            momentos = momentos_observaveis_locais(estado, dims, modos, posicoes)
            dados.append([momentos[medida] for medida in medidas])

        dados = np.asarray(dados)
        residuos = np.asarray(residuos)
        validos = np.isfinite(residuos) & (residuos < tolerancia)
        for eixo, medida, valores in zip(eixos, medidas, dados.T):
            curva, = eixo.semilogx(epsilons, valores, ".-", label=f"{g:g}".replace(".", ","))
            eixo.axhline(referencia[medida], color=curva.get_color(), linestyle="--", alpha=0.6)
            if not validos.all():
                eixo.scatter(epsilons[~validos], valores[~validos], marker="x", color=curva.get_color())
        print(f"g={g:g} | dentro da tolerância: {validos.sum()}/{len(epsilons)} "
              f"| resíduo máximo: {residuos.max():.2e} "
              f"| Pearson fundamental: {referencia['pearson']:.4f} "
              f"| Pearson em ε=1: {dados[-1, 1]:.4f}")

    for eixo, rotulo in zip(eixos, ("Covariância de posição", "Correlação de Pearson")):
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.25)
    eixos[0].legend(title="Acoplamento g", ncols=2)
    figura.suptitle("Correlações de posição para diferentes acoplamentos\n"
                    r"Tracejado: estado fundamental · $\times$: resíduo $\geq 10^{-6}$")
    figura.supxlabel(r"Regularização $\varepsilon$")
    DESTINO.mkdir(exist_ok=True)
    salvar(figura, "correlacoes_por_acoplamento")


if __name__ == "__main__":
    main()
