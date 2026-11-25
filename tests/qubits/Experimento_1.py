"""Compara reconstruções de uma cadeia aberta com marginais locais I₂/2."""

import argparse
from functools import partial
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulos.Hamiltoniano_ex import (
    PAULI_X, PAULI_Y, PAULI_Z, Hamiltoniano_XY, Hamiltoniano_ZZ, dimensoes_qubits,
)
from modulos.dual_entropico import DualEntropico
from modulos.estados import entropia_von_neumann, estado_gibbs, negatividade
from modulos.marginais import restricoes_lineares
from modulos.operadores import momentos_observaveis_locais
from modulos.otimizacao import varrer_dual
from modulos.validacao import comparar_marginais

RHO_LOCAL = np.eye(2) / 2
MEDIDAS = {
    "entropia": "Entropia (nats)",
    "energia": "Energia média ⟨H⟩",
    "negatividade": "Negatividade (1 | restante)",
}


def _preparar_cadeia(J, N):
    dims = dimensoes_qubits(N)
    marginais = {(i,): RHO_LOCAL for i in range(N)}
    Q, q = restricoes_lineares(marginais, dims)
    casos = {
        "Sem interação": np.zeros((2**N, 2**N)),
        "Ising ZZ": Hamiltoniano_ZZ(J, N),
        "Interação XY": Hamiltoniano_XY(J, N),
    }
    return dims, Q, q, casos


def _varrer_caso(nome, H, Q, q, epsilons, *, exigir_convergencia=True):
    energia_minima = np.linalg.eigvalsh(H)[0]
    resultados = varrer_dual(
        partial(DualEntropico, H, Q, q), epsilons,
        fator_learnrate=0.2, tolerancia=1e-10, maxiter=5000,
        u_inicial=lambda problema: _multiplicadores_iniciais(problema, energia_minima),
    )
    for epsilon, resultado in zip(epsilons, resultados):
        if exigir_convergencia and not resultado["convergiu"]:
            raise RuntimeError(
                f"{nome}, ε={epsilon:g}: otimização não convergiu "
                f"(resíduo={resultado['norma_residuos']:.3e})."
            )
        yield resultado


def _multiplicadores_iniciais(problema, energia_minima):
    # Inicializa o traço em no máximo 1, inclusive para ε pequeno.
    u_inicial = np.zeros(len(problema.q))
    u_inicial[0] = energia_minima - problema.epsilon * np.log(problema.H.shape[0])
    return u_inicial


def _medir(estado, H):
    return {
        "entropia": entropia_von_neumann(estado),
        "energia": np.trace(estado @ H).real,
        "negatividade": negatividade(estado),
    }


def executar_caso(nome, H, Q, q, epsilon, dims):
    """Imprime o diagnóstico de um Hamiltoniano e retorna o estado reconstruído."""
    resultado = next(_varrer_caso(nome, H, Q, q, (epsilon,), exigir_convergencia=False))
    gamma = resultado["estado"]
    referencia = estado_gibbs(H, beta=1 / epsilon)
    erro = np.linalg.norm(gamma - referencia, ord="fro")
    residuos = resultado["norma_residuos"]
    pureza = np.trace(gamma @ gamma).real
    medidas = _medir(gamma, H)
    comparacao = comparar_marginais(gamma, {(i,): RHO_LOCAL for i in range(len(dims))}, dims)
    erros_marginais = [dados["frobenius_error"] for dados in comparacao.values()]
    print(
        f"{nome:16s} eps={epsilon:.3g}  erro={erro:.3e}  "
        f"resíduo={residuos:.3e}  pureza={pureza:.6f}  "
        f"entropia={medidas['entropia']:.6f} nats  negatividade={medidas['negatividade']:.6f}"
    )
    if len(dims) == 2:
        erro_a, erro_b = erros_marginais
        print(f"  Erro marginal A={erro_a:.3e}  Erro marginal B={erro_b:.3e}")
    else:
        print(f"  Maior erro entre as {len(dims)} marginais={max(erros_marginais):.3e}")
    if not resultado["convergiu"] or erro > 1e-5:
        print("  Aviso: confira a convergência e os resíduos do otimizador.")
    print("  Estado reconstruído (base computacional):")
    print(np.array2string(gamma, precision=4, suppress_small=True))
    return gamma


def varrer_epsilon(valores_epsilon=None, J=1.0, N=2):
    """Retorna a figura e um dicionário de métricas em ordem crescente de ε."""
    import matplotlib.pyplot as plt

    if valores_epsilon is None:
        valores_epsilon = np.linspace(0.1, 2.0, 41)
    epsilons = np.asarray(valores_epsilon, dtype=float)
    if (
        epsilons.ndim != 1
        or epsilons.size == 0
        or not np.isfinite(epsilons).all()
        or np.any(epsilons <= 0)
    ):
        raise ValueError("valores_epsilon deve ser um vetor não vazio de valores positivos e finitos.")
    epsilons = np.sort(epsilons)

    _, Q, q, casos = _preparar_cadeia(J, N)
    dados = {"epsilon": epsilons}
    for nome, H in casos.items():
        dados[nome] = {medida: np.empty(epsilons.size) for medida in MEDIDAS}
        for i, resultado in enumerate(_varrer_caso(nome, H, Q, q, epsilons)):
            for medida, valor in _medir(resultado["estado"], H).items():
                dados[nome][medida][i] = valor

    figura, eixos = plt.subplots(3, 1, sharex=True, figsize=(8, 8), layout="constrained")
    figura.suptitle(f"Experimento 1 — N = {N}, J = {J:g}")
    for eixo, (medida, rotulo) in zip(eixos, MEDIDAS.items()):
        for nome, estilo in zip(casos, ("-", "--", "-.")):
            eixo.plot(epsilons, dados[nome][medida], linestyle=estilo, label=nome)
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.25)
    eixos[0].legend()
    eixos[-1].set_xlabel("Regularização ε")
    return figura, dados


def gerar_figuras_slides():
    """Grava e retorna os caminhos dos seis PNGs do Beamer para dois qubits."""
    import matplotlib.pyplot as plt
    from tests.qubits.figuras import matriz_densidade, salvar

    dims, Q, q, casos = _preparar_cadeia(J=1, N=2)
    epsilons = np.unique(np.r_[np.geomspace(0.1, 3, 41), 0.5])
    nomes_png = ("caso_I_sem_interacao", "caso_II_ising_zz", "caso_III_xy")
    caminhos = []
    correlacoes = {}
    figura_erro, eixo_erro = plt.subplots(figsize=(5.6, 4.3), layout="constrained")
    figura_medidas, eixos_medidas = plt.subplots(
        2, 1, figsize=(5.6, 4.3), sharex=True, layout="constrained",
    )

    for (nome, H), nome_png in zip(casos.items(), nomes_png):
        erros, entropias, negatividades, covariancias = [], [], [], []
        for epsilon, resultado in zip(epsilons, _varrer_caso(nome, H, Q, q, epsilons)):
            estado = resultado["estado"]
            referencia = estado_gibbs(H, beta=1 / epsilon)
            erro = np.linalg.norm(estado - referencia, ord="fro")
            neg = negatividade(estado)
            erros.append(erro)
            entropias.append(entropia_von_neumann(estado))
            negatividades.append(neg)
            covariancias.append([
                momentos_observaveis_locais(estado, dims, (0, 1), (pauli, pauli))["covariancia"]
                for pauli in (PAULI_X, PAULI_Y, PAULI_Z)
            ])
            if epsilon == 0.5:
                figura, eixos = plt.subplots(1, 3, figsize=(9, 2.6), layout="constrained")
                matriz_densidade(eixos[0], estado, "Reconstrução")
                matriz_densidade(eixos[1], referencia, "Referência de Gibbs")
                eixos[2].axis("off")
                resumo = (
                    f"J = 1    ε = 0,5\n\nErro de Frobenius: {erro:.2e}\n"
                    f"Norma do resíduo: {resultado['norma_residuos']:.2e}\n"
                    f"Negatividade: {neg:.4f}"
                )
                eixos[2].text(0, 0.5, resumo.replace(".", ","), va="center", linespacing=1.5)
                caminhos.append(salvar(figura, nome_png, dpi=300))

        eixo_erro.semilogx(epsilons, erros, ".-", label=nome)
        eixos_medidas[0].semilogx(epsilons, entropias, ".-", label=nome)
        eixos_medidas[1].semilogx(epsilons, negatividades, ".-", label=nome)
        correlacoes[nome] = np.asarray(covariancias)

    eixo_erro.set_ylabel("Erro de Frobenius")
    eixo_erro.set_yscale("symlog", linthresh=1e-13)
    eixo_erro.set_yticks([0, 1e-12, 1e-11, 1e-10],
                         ["0", r"$10^{-12}$", r"$10^{-11}$", r"$10^{-10}$"])
    eixo_erro.legend(loc="lower right")
    eixos_medidas[0].set_ylabel("Entropia (nats)")
    eixos_medidas[0].legend(loc="center left", fontsize=9)
    eixos_medidas[1].set_ylabel("Negatividade")
    limiar = 2 / np.log(1 + np.sqrt(2))
    linha = eixos_medidas[1].axvline(
        limiar, color="0.4", linestyle="--", label=r"$\varepsilon_c\approx 2{,}27$",
    )
    eixos_medidas[1].legend(handles=[linha], loc="center left", fontsize=9)

    figura_pauli, eixos_pauli = plt.subplots(
        1, 2, figsize=(10, 4.3), sharex=True, sharey=True, layout="constrained",
    )
    figura_pauli.suptitle("Correlações de Pauli · dois qubits, J = 1")
    for eixo, nome in zip(eixos_pauli, ("Ising ZZ", "Interação XY")):
        eixo.set_title(nome)
        xx, yy, zz = correlacoes[nome].T
        # As interações preservam a simetria entre X e Y.
        np.testing.assert_allclose(xx, yy, atol=1e-10)
        sem_interacao = correlacoes["Sem interação"][:, 2]
        if nome == "Ising ZZ":
            np.testing.assert_allclose(xx, sem_interacao, atol=1e-10)
            eixo.semilogx(epsilons, sem_interacao, "--", color="C0",
                         label=r"Sem interação e $C_{xx}=C_{yy}$")
        else:
            eixo.semilogx(epsilons, sem_interacao, "--", color="C0", label="Sem interação")
            eixo.semilogx(epsilons, xx, ".-", color="C3", label=r"$C_{xx}=C_{yy}$")
        eixo.semilogx(epsilons, zz, ".-", color="C1" if nome == "Ising ZZ" else "C2",
                     label=r"$C_{zz}$")
        eixo.set_ylim(-1.07, 1.07)
        eixo.legend(loc="center left", fontsize=10)
    eixos_pauli[0].set_ylabel("Covariância das medidas de Pauli")

    for figura, eixos, nome in (
        (figura_erro, (eixo_erro,), "comparacao_erro_reconstrucao"),
        (figura_medidas, eixos_medidas, "comparacao_entropia_negatividade"),
        (figura_pauli, eixos_pauli, "correlacoes_pauli"),
    ):
        for eixo in eixos:
            eixo.grid(alpha=0.25)
            eixo.set_xlim(0.09, 3.35)
        figura.supxlabel(r"Regularização $\varepsilon$")
        caminhos.append(salvar(figura, nome))
    return caminhos


def main(N=2, J=1.0, epsilon=0.5):
    """Imprime a comparação com o estado de Gibbs para um valor de ε."""
    dims, Q, q, casos = _preparar_cadeia(J, N)
    print(f"Parâmetros: N={N}, J={J:g}, epsilon={epsilon:g}, rho_i=I_2/2\n")
    for nome, H in casos.items():
        executar_caso(nome, H, Q, q, epsilon, dims)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=2, help="Número de qubits da cadeia (padrão: 2).")
    saida = parser.add_mutually_exclusive_group()
    saida.add_argument("--figuras-slides", action="store_true", help="Grava os PNGs do Beamer.")
    saida.add_argument(
        "--varrer-epsilon", type=Path, metavar="ARQUIVO",
        help="Salva o gráfico da varredura em ε.",
    )
    args = parser.parse_args()
    if args.qubits < 2:
        parser.error("--qubits deve ser maior ou igual a 2.")
    if args.figuras_slides:
        if args.qubits != 2:
            parser.error("--figuras-slides requer dois qubits.")
        for caminho in gerar_figuras_slides():
            print(caminho)
    elif args.varrer_epsilon is None:
        main(N=args.qubits)
    else:
        figura, _ = varrer_epsilon(N=args.qubits)
        figura.savefig(args.varrer_epsilon, dpi=180)
