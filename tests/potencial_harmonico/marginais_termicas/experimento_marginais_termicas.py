"""Reconstrói um estado de osciladores a partir de marginais térmicas."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
ROOT = PASTA_EXPERIMENTO.parents[2]
sys.path.insert(0, str(ROOT))

from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import entropia_von_neumann, estado_termico_oscilador, tensor
from modulos.marginais import restricoes_lineares
from modulos.operadores import momentos_observaveis_locais, operador_posicao
from modulos.otimizacao import varrer_dual
from modulos.validacao import comparar_marginais
from modulos.visualizacao import visualizar_marginais, visualizar_posicoes_conjuntas


def main():
    dims = (4, 4)
    omega = 1.0
    g = 0.5
    temperatura_local = 5
    epsilons = np.geomspace(5, 1, 10)
    indice_visualizacao = 0
    pares_interacao = [(0, 1, g)]
    marginais = {
        (i,): estado_termico_oscilador(dim, temperatura=temperatura_local, omega=omega)
        for i, dim in enumerate(dims)
    }
    H = hamiltoniano_bilinear(dims, omega, pares_interacao)
    Q, q = restricoes_lineares(marginais, dims)
    estado_produto = tensor(*(marginais[(i,)] for i in range(len(dims))))
    energia_produto = np.einsum("ij,ji->", H, estado_produto).real
    sistemas_posicao = (0, 1)
    posicoes = tuple(operador_posicao(dims[i]) for i in sistemas_posicao)
    momentos_produto = momentos_observaveis_locais(estado_produto, dims, sistemas_posicao, posicoes)
    medidas = (
        ("Energia ⟨H⟩", energia_produto),
        (r"Diferença de energia $|E - E_{\mathrm{prod}}|$", None),
        ("Entropia (nats)", entropia_von_neumann(estado_produto)),
        (r"Distância $\|\Gamma - \Gamma_{\mathrm{prod}}\|_F$", None),
        ("Norma dos resíduos", None),
        ("Covariância de posição", momentos_produto["covariancia"]),
        ("Correlação de Pearson", momentos_produto["pearson"]),
    )
    dados = []
    marginais_por_epsilon = []
    estados = []
    erros_marginais = {chave: [] for chave in marginais}
    resultados = varrer_dual(
        lambda epsilon: DualEntropico(H, Q, q, epsilon), epsilons,
        fator_learnrate=0.02, reutilizar_multiplicadores=True,
        tolerancia=1e-10, maxiter=5000,
    )
    for epsilon, resultado in zip(epsilons, resultados):
        estado = resultado["estado"]
        estados.append(estado)
        comparacao = comparar_marginais(estado, marginais, dims)
        marginais_por_epsilon.append({
            chave: dados["calculated_marginal"] for chave, dados in comparacao.items()
        })
        for chave, diagnostico in comparacao.items():
            erros_marginais[chave].append(diagnostico["frobenius_error"])
        energia = np.einsum("ij,ji->", H, estado).real
        momentos = momentos_observaveis_locais(estado, dims, sistemas_posicao, posicoes)
        dados.append((
            energia, abs(energia - energia_produto), entropia_von_neumann(estado),
            np.linalg.norm(estado - estado_produto, ord="fro"), resultado["norma_residuos"],
            momentos["covariancia"], momentos["pearson"],
        ))
        print(f"ε={epsilon:g} | convergiu={resultado['convergiu']} | resíduos={resultado['norma_residuos']:.3e}")

    figura, eixos = plt.subplots(4, 2, sharex=True, figsize=(11, 10), layout="constrained")
    figura.suptitle(f"Reconstrução com marginais térmicas (T = {temperatura_local:g})")
    for eixo, (rotulo, referencia), valores in zip(eixos.flat, medidas, np.asarray(dados).T):
        eixo.semilogx(epsilons, valores, marker=".")
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.25)
        if referencia is not None:
            eixo.axhline(referencia, color="0.4", linestyle="--", label="Estado produto")
            eixo.legend()
    eixo_erros = eixos[-1, 1]
    for (i,), estilo in zip(marginais, ("-", "--")):
        eixo_erros.semilogx(epsilons, erros_marginais[(i,)], linestyle=estilo, label=f"Oscilador {i}")
    eixo_erros.set_ylabel(r"Erro da marginal $\|\hat{\rho}_i - \rho_i\|_F$")
    eixo_erros.grid(alpha=0.25)
    eixo_erros.legend()
    figura.supxlabel(r"Regularização $\epsilon$")

    epsilon_visualizacao = epsilons[indice_visualizacao]
    estado = estados[indice_visualizacao]
    reconstruidas = marginais_por_epsilon[indice_visualizacao]

    figura, _ = visualizar_marginais(marginais)
    figura.suptitle("Marginais térmicas prescritas")
    figura.savefig(PASTA_EXPERIMENTO / "marginais_potencial_harmonico.png", dpi=150)

    figura_reconstruida, _ = visualizar_marginais(reconstruidas)
    figura_reconstruida.suptitle(f"Marginais reconstruídas (ε = {epsilon_visualizacao:g})")
    figura_reconstruida.savefig(PASTA_EXPERIMENTO / "marginais_reconstruidas_potencial_harmonico.png", dpi=150)

    grade = np.linspace(-5.0, 5.0, 41)
    pearson = momentos_observaveis_locais(estado, dims, sistemas_posicao, posicoes)["pearson"]
    visualizar_posicoes_conjuntas(
        estado, grade, sistemas_posicao, dims,
        titulo=f"Distribuição conjunta de posições (ε = {epsilon_visualizacao:g})\n"
               f"Correlação de Pearson = {pearson:.4f}",
    )
    plt.show()


if __name__ == "__main__":
    main()
