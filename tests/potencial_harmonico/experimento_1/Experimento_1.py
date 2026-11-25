"""Reconstrói o estado fundamental de dois osciladores a partir das marginais."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Permite executar o arquivo diretamente ou via python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modulos.diagnosticos_reconstrucao import cota_frobenius, gap_espectral
from modulos.estados import entropia_von_neumann
from modulos.visualizacao import visualizar_marginais, visualizar_posicoes_conjuntas
from tests.potencial_harmonico.experimento_1.base import (
    DIMENSOES, FATOR_LEARNRATE, MAXITER, METODO, MODOS, OMEGA, PASTA, TOLERANCIA,
    ExperimentoHarmonico,
)

from tests.potencial_harmonico.experimento_1.figuras import salvar


def _acrescentar(caminho, titulo, campos):
    linhas = [f"\n{titulo}\n"]
    for nome, valor in campos.items():
        if isinstance(valor, np.ndarray):
            matriz = np.array2string(valor, floatmode="unique", threshold=valor.size)
            linhas.append(f"{nome} =\n{matriz}\n")
        else:
            linhas.append(f"{nome} = {valor}\n")
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write("".join(linhas))


def iniciar_protocolo(pasta, nome, titulo, configuracao, epsilons, marginais, **campos):
    """Acrescenta o cabeçalho de uma execução ao arquivo da configuração fixa."""
    assinatura = json.dumps(configuracao, sort_keys=True)
    identificador = hashlib.sha256(assinatura.encode("utf-8")).hexdigest()[:16]
    dimensoes = "x".join(map(str, configuracao["dims"]))
    omega = configuracao["omega"]
    pasta.mkdir(exist_ok=True)
    caminho = pasta / f"{nome}_{dimensoes}_omega-{omega:g}_{identificador}.txt"
    _acrescentar(caminho, f"=== {titulo} | {datetime.now(timezone.utc).isoformat()} ===", {
        **configuracao,
        "epsilons_em_ordem_de_execucao": np.asarray(epsilons),
        **campos,
        **{f"marginal_prescrita_{sistema}": valor for sistema, valor in marginais.items()},
    })
    return caminho


def registrar_resultado(caminho, resultado, marginais_reconstruidas, **campos):
    """Acrescenta as métricas, a saída do otimizador e as marginais de um ε."""
    _acrescentar(caminho, "--- Resultado ---", {
        **campos,
        **resultado,
        "historico": np.asarray(resultado["historico"]),
        **{
            f"marginal_reconstruida_{sistema}": valor
            for sistema, valor in marginais_reconstruidas.items()
        },
    })


def figura_frobenius():
    """Grava a varredura da distância de Frobenius e retorna o caminho do PNG."""
    experimento = ExperimentoHarmonico()
    gap = gap_espectral(experimento.H)

    epsilons = np.geomspace(0.01, 1, 21)
    distancias = np.empty_like(epsilons)
    convergiu = np.empty(epsilons.size, dtype=bool)
    for i, resultado in enumerate(experimento.varrer(epsilons, "logaritmico")):
        distancias[i] = np.linalg.norm(resultado["estado"] - experimento.estado, ord="fro")
        convergiu[i] = resultado["convergiu"]

    figura, eixo = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
    eixo.loglog(epsilons, distancias, ".-", label="Reconstrução")
    eixo.loglog(
        epsilons, cota_frobenius(epsilons, OMEGA, experimento.marginais), "--",
        label=rf"Estimativa com $\omega={OMEGA:g}$",
    )
    eixo.loglog(
        epsilons, cota_frobenius(epsilons, gap, experimento.marginais), ":",
        label=rf"Cota com o gap $\Delta={gap:.3f}$",
    )
    if not convergiu.all():
        eixo.scatter(
            epsilons[~convergiu], distancias[~convergiu], marker="x", color="C0",
            label=rf"Resíduo $\geq {TOLERANCIA:g}$",
        )
    eixo.set(
        xlabel=r"Regularização $\varepsilon$",
        ylabel=r"Distância $\|\Gamma_\varepsilon-\Gamma_0\|_F$",
        title="Reconstrução do estado fundamental",
    )
    eixo.grid(alpha=0.25)
    eixo.legend()
    return salvar(figura, "distancia_frobenius")


def main():
    """Exibe as figuras da reconstrução e acrescenta os dados ao protocolo."""
    epsilons = np.array([0.1])
    epsilon_visualizacao = 0.1
    experimento = ExperimentoHarmonico()
    configuracao = {
        "dims": DIMENSOES,
        "omega": OMEGA,
        "pares_interacao": experimento.pares_interacao,
        "metodo": METODO,
        "tolerancia": TOLERANCIA,
        "maxiter": MAXITER,
        "fator_learnrate": FATOR_LEARNRATE,
    }
    H = experimento.H
    energia_fundamental, gamma0 = experimento.energia, experimento.estado
    marginais = experimento.marginais
    momentos_fundamental = experimento.momentos_posicao(gamma0)
    correlacao_fundamental = momentos_fundamental["covariancia"]
    pearson_fundamental = momentos_fundamental["pearson"]
    caminho_protocolo = iniciar_protocolo(
        PASTA / "protocolos", "Experimento_1", "Experimento 1",
        configuracao, epsilons, marginais,
        inicializacao="Zero no primeiro ε, multiplicadores anteriores nos demais",
        regra_learnrate="fator_learnrate * epsilon",
        energia_fundamental=energia_fundamental,
        correlacao_fundamental=correlacao_fundamental,
        pearson_fundamental=pearson_fundamental,
        Hamiltoniano=H, estado_fundamental=gamma0,
        Q=np.asarray(experimento.Q), q=np.asarray(experimento.q),
    )
    medidas = (
        ("Energia ⟨H⟩", energia_fundamental),
        ("Erro de energia |E − E₀|", None),
        ("Entropia (nats)", None),
        ("Distância ‖Γ − Γ₀‖₍F₎", None),
        ("Norma dos resíduos", None),
        ("Covariância de posição", correlacao_fundamental),
        ("Correlação de Pearson", pearson_fundamental),
    )
    dados = []
    u_inicial = None
    estado_visualizacao = None
    resultados = experimento.varrer(epsilons, "exponencial", reutilizar_multiplicadores=True)
    for epsilon, resultado in zip(epsilons, resultados):
        learnrate = FATOR_LEARNRATE * epsilon
        multiplicadores_iniciais = np.zeros(len(experimento.Q)) if u_inicial is None else u_inicial.copy()
        u_inicial = resultado["multiplicadores"]
        gamma = resultado["estado"]
        energia_reconstruida = np.trace(H @ gamma).real
        erro_energia = abs(energia_reconstruida - energia_fundamental)
        entropia = entropia_von_neumann(gamma)
        distancia = np.linalg.norm(gamma - gamma0, ord="fro")
        norma_residuos = resultado["norma_residuos"]
        momentos = experimento.momentos_posicao(gamma)
        correlacao_reconstruida = momentos["covariancia"]
        pearson_reconstruido = momentos["pearson"]
        if epsilon == epsilon_visualizacao:
            estado_visualizacao = gamma, pearson_reconstruido
        dados.append((
            energia_reconstruida, erro_energia, entropia,
            distancia, norma_residuos, correlacao_reconstruida, pearson_reconstruido,
        ))
        marginais_reconstruidas = experimento.calcular_marginais(gamma)
        registrar_resultado(
            caminho_protocolo, resultado, marginais_reconstruidas,
            epsilon=epsilon, learnrate=learnrate, multiplicadores_iniciais=multiplicadores_iniciais,
            energia_reconstruida=energia_reconstruida, erro_energia=erro_energia,
            entropia=entropia, distancia_frobenius=distancia,
            correlacao_reconstruida=correlacao_reconstruida, pearson_reconstruido=pearson_reconstruido,
        )

        print(
            f"epsilon={epsilon:.3e} | "
            f"E0={energia_fundamental:.10f} | "
            f"E={energia_reconstruida:.10f} | "
            f"|E-E0|={erro_energia:.3e} | "
            f"S={entropia:.8f} | "
            f"||Gamma-Gamma0||_F={distancia:.3e} | "
            f"||resíduos||={norma_residuos:.3e} | "
            f"convergiu={resultado['convergiu']} | "
            f"Cov(X0, X1)={correlacao_reconstruida:.8f} "
            f"(fundamental: {correlacao_fundamental:.8f}) | "
            f"Pearson={pearson_reconstruido:.8f} "
            f"(fundamental: {pearson_fundamental:.8f})"
        )

    print(f"Protocolo salvo em {caminho_protocolo}")

    figura, eixos = plt.subplots(4, 2, sharex=True, figsize=(11, 10), layout="constrained")
    figura.delaxes(eixos[-1, 1])
    eixos[-2, 1].tick_params(labelbottom=True)
    figura.suptitle("Reconstrução do estado fundamental")
    for eixo, (rotulo, referencia), valores in zip(eixos.flat, medidas, np.asarray(dados).T):
        eixo.semilogx(epsilons, valores, marker=".")
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.25)
        if referencia is not None:
            eixo.axhline(referencia, color="0.4", linestyle="--", label="Estado fundamental")
            eixo.legend()
    figura.supxlabel(r"Regularização $\epsilon$")

    figura, _ = visualizar_marginais(marginais)
    figura.suptitle("Marginais prescritas do estado fundamental")
    figura, _ = visualizar_marginais(marginais_reconstruidas)
    figura.suptitle(f"Marginais do estado reconstruído (ε = {epsilons[-1]:g})")

    if estado_visualizacao is not None:
        gamma_visualizacao, pearson_visualizacao = estado_visualizacao
        grade = np.linspace(-5.0, 5.0, 41)
        visualizar_posicoes_conjuntas(
            gamma_visualizacao, grade, MODOS, DIMENSOES,
            titulo=f"Distribuição conjunta de posições (ε = {epsilon_visualizacao:g})\n"
                   f"Correlação de Pearson = {pearson_visualizacao:.4f}",
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figura-frobenius", action="store_true",
        help="Grava o PNG com a varredura e as estimativas de Frobenius.",
    )
    if parser.parse_args().figura_frobenius:
        plt.switch_backend("Agg")
        print(figura_frobenius())
    else:
        main()
