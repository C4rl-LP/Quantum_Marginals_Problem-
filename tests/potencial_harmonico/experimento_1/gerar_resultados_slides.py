"""Gera as figuras e os frames Beamer dos resultados do Experimento 1."""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[2]))

from modulos.dual_entropico import DualEntropico
from modulos.estados import entropia_von_neumann, estado_fundamental
from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.marginais import restricoes_lineares
from modulos.operadores import momentos_observaveis_locais, operador_posicao, partial_trace
from modulos.otimizacao import otimizar_dual
from modulos.visualizacao import distribuicao_conjunta_posicoes, wigner_marginal
from tests.potencial_harmonico.experimento_1.Experimento_1 import dims, omega, pares_interacao

DESTINO = PASTA / "figuras_slides"


def salvar(figura, nome):
    figura.savefig(DESTINO / f"{nome}.png", dpi=200)
    plt.close(figura)


def mapas(grade, dados, rotulos, eixos, grandeza, arquivo, colunas):
    figura, paineis = plt.subplots(
        len(dados) // colunas, colunas, figsize=(3.2 * colunas + 0.5, 2.75 * (len(dados) // colunas)),
        layout="constrained", sharex=True, sharey=True,
    )
    escala = Normalize(min(d.min() for d in dados), max(d.max() for d in dados))
    for eixo, valores, rotulo in zip(paineis.flat, dados, rotulos):
        imagem = eixo.pcolormesh(grade, grade, valores, shading="auto", norm=escala, cmap="viridis")
        eixo.set_title(rotulo)
        eixo.set_aspect("equal")
    figura.supxlabel(eixos[0])
    figura.supylabel(eixos[1])
    figura.colorbar(imagem, ax=paineis, label=grandeza, shrink=0.85)
    salvar(figura, arquivo)


def numero(valor):
    mantissa, expoente = f"{valor:.2e}".split("e")
    return rf"{mantissa.replace('.', '{,}')}\times10^{{{int(expoente)}}}"


def main():
    DESTINO.mkdir(exist_ok=True)
    H = hamiltoniano_bilinear(dims, omega, pares_interacao)
    E0, gamma0 = estado_fundamental(H)
    sistemas = (0, 1)
    marginais = {(i,): partial_trace(gamma0, keep=(i,), dims=dims) for i in sistemas}
    Q, q = restricoes_lineares(marginais, dims)
    posicoes = tuple(operador_posicao(dims[i]) for i in sistemas)
    pearson0 = momentos_observaveis_locais(gamma0, dims, sistemas, posicoes)["pearson"]
    epsilons = np.geomspace(1, 0.01, 21)
    tolerancia = 1e-6
    dados = []
    estados = []
    for epsilon in epsilons:
        problema = DualEntropico(H, Q, q, epsilon, forma="logaritmico")
        resultado = otimizar_dual(
            problema, learnrate=0.1 * epsilon, tolerancia=tolerancia, maxiter=10000,
        )
        gamma = resultado["estado"]
        erros_locais = [
            np.linalg.norm(partial_trace(gamma, keep=chave, dims=dims) - alvo, ord="fro")
            for chave, alvo in marginais.items()
        ]
        momentos = momentos_observaveis_locais(gamma, dims, sistemas, posicoes)
        dados.append([
            np.trace(H @ gamma).real - E0, entropia_von_neumann(gamma),
            np.linalg.norm(gamma - gamma0, ord="fro"), resultado["norma_residuos"],
            *erros_locais, momentos["covariancia"], momentos["pearson"],
        ])
        estados.append(gamma)

    dados = np.asarray(dados)
    assert np.isfinite(dados).all()
    figura, eixos = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True, layout="constrained")
    for eixo, indice, rotulo in zip(
        eixos.flat, (0, 1, 2, 3, 4, 7),
        (r"$E-E_0$", "Entropia (nats)", r"$\|\Gamma-\Gamma_0\|_F$",
         "Norma dos resíduos", "Erro de cada marginal", "Pearson de posição"),
    ):
        eixo.semilogx(epsilons, dados[:, indice], ".-", label="Modo A" if indice == 4 else None)
        eixo.set_title(rotulo)
        eixo.grid(alpha=0.25)
    eixos[1, 0].set_yscale("log")
    eixos[1, 0].axhline(tolerancia, color="0.4", ls="--", label="Tolerância")
    eixos[1, 0].legend()
    eixos[1, 1].loglog(epsilons, dados[:, 5], "--", label="Modo B")
    eixos[1, 1].legend()
    eixos[1, 2].axhline(pearson0, color="0.4", ls="--", label="Referência")
    eixos[1, 2].legend()
    figura.supxlabel(r"Regularização $\varepsilon$")
    salvar(figura, "varredura")

    grade = np.linspace(-3.5, 3.5, 101)
    amostras = (gamma0, estados[0], estados[10], estados[-1])
    titulos = ("Referência", r"$\varepsilon=1$", r"$\varepsilon=0{,}1$", r"$\varepsilon=0{,}01$")
    mapas(
        grade, [wigner_marginal(gamma, grade, grade, keep=(i,), dims=dims)
                for i in sistemas for gamma in (gamma0, estados[0], estados[-1])],
        [f"Modo {modo} · {titulos[j]}" for modo in ("A", "B") for j in (0, 1, 3)],
        ("Posição", "Momento"), "Wigner", "wigner_marginais", 3,
    )
    mapas(
        grade, [distribuicao_conjunta_posicoes(gamma, grade, sistemas, dims).T for gamma in amostras],
        titulos, (r"$x_A$", r"$x_B$"), "Densidade de posição", "posicoes_conjuntas", 4,
    )
    escrever_frames(dados, E0, pearson0, tolerancia)


def escrever_frames(dados, E0, pearson0, tolerancia):
    linhas = []
    for indice, epsilon in zip((0, 10, 20), ("1", "0{,}1", "0{,}01")):
        celulas = " & ".join(f"${numero(dados[indice, j])}$" for j in (0, 2, 3, 7))
        linhas.append(rf"${epsilon}$ & {celulas} \\")
    tabela = "\n".join(linhas)
    convergiram = np.count_nonzero(dados[:, 3] < tolerancia)
    texto = r"""
% Copie a pasta figuras_slides para junto do arquivo principal do Beamer.
\begin{frame}{Reconstrução do estado fundamental: varredura}
\centering
\includegraphics[width=\linewidth,height=.69\textheight,keepaspectratio]{figuras_slides/varredura.png}

\small
Dual logarítmico, 21 valores de $\varepsilon$ entre $1$ e $0{,}01$,
multiplicadores inicializados em zero em cada ponto, $\eta=0{,}1\varepsilon$ e até 10.000 iterações.
\end{frame}

\begin{frame}{Reconstrução do estado fundamental: resultados}
\centering
\small
\begin{tabular}{ccccc}
\hline
$\varepsilon$ & $E-E_0$ & $\|\Gamma-\Gamma_0\|_F$ &
$\|r\|_2$ & Pearson \\
\hline
@TABELA@
\hline
\end{tabular}

\medskip
Referência numérica: $E_0=@ENERGIA@$ e Pearson $=@PEARSON@$.

\medskip
@CONVERGIRAM@ dos 21 pontos atingiram a tolerância de $10^{-6}$ no resíduo.
\end{frame}

\begin{frame}{Wigner das marginais reconstruídas}
\centering
\includegraphics[width=\linewidth,height=.81\textheight,keepaspectratio]{figuras_slides/wigner_marginais.png}

\small Os seis painéis usam a mesma escala de cores.
\end{frame}

\begin{frame}{Distribuição conjunta de posições}
\centering
\includegraphics[width=\linewidth,height=.75\textheight,keepaspectratio]{figuras_slides/posicoes_conjuntas.png}

\small Comparação entre a referência e três valores de $\varepsilon$, com escala de cores comum.
\end{frame}
"""
    texto = texto.replace("@TABELA@", tabela).replace("@CONVERGIRAM@", str(convergiram))
    texto = texto.replace("@ENERGIA@", f"{E0:.6f}".replace(".", "{,}"))
    texto = texto.replace("@PEARSON@", f"{pearson0:.6f}".replace(".", "{,}"))
    (PASTA / "resultados_experimentais.tex").write_text(texto.lstrip(), encoding="utf-8")


if __name__ == "__main__":
    main()
