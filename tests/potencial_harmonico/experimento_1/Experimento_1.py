"""Reconstrói o estado fundamental de dois osciladores a partir das marginais."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Parâmetros do experimento: altere estes valores para modificar o exemplo.
dims = (4, 4)
omega = 1.0
g = 0.5
epsilons = np.geomspace(0.1, 0.1, 1)  # inclui epsilon = 0.1
epsilon_visualizacao = .1
pares_interacao = [(0, 1, g)]

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
ROOT = PASTA_EXPERIMENTO.parents[2]
sys.path.insert(0, str(ROOT))

from modulos.Hamiltoniano_ex import hamiltoniano_bilinear
from modulos.dual_entropico import DualEntropico
from modulos.estados import entropia_von_neumann, estado_fundamental
from modulos.marginais import restricoes_lineares
from modulos.operadores import momentos_observaveis_locais, operador_posicao, partial_trace
from modulos.otimizacao import otimizar_dual
from modulos.visualizacao import distribuicao_conjunta_posicoes, visualizar_marginais


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


if __name__ == "__main__":
    configuracao = {
        "dims": dims,
        "omega": omega,
        "pares_interacao": pares_interacao,
        "metodo": "gradient",
        "tolerancia": 1e-6,
        "maxiter": 10000,
        "fator_learnrate": 0.1,
    }
    H = hamiltoniano_bilinear(dims, omega, pares_interacao)
    energia_fundamental, gamma0 = estado_fundamental(H)

    marginais = {
        (i,): partial_trace(gamma0, keep=(i,), dims=dims)
        for i in range(len(dims))
    }
    Q, q = restricoes_lineares(marginais, dims)
    sistemas_posicao = (0, 1)
    posicoes = tuple(operador_posicao(dims[i]) for i in sistemas_posicao)
    momentos_fundamental = momentos_observaveis_locais(gamma0, dims, sistemas_posicao, posicoes)
    correlacao_fundamental = momentos_fundamental["covariancia"]
    pearson_fundamental = momentos_fundamental["pearson"]
    caminho_protocolo = iniciar_protocolo(
        PASTA_EXPERIMENTO / "protocolos", "Experimento_1", "Experimento 1",
        configuracao, epsilons, marginais,
        inicializacao="Zero no primeiro ε, multiplicadores anteriores nos demais",
        regra_learnrate="fator_learnrate * epsilon",
        energia_fundamental=energia_fundamental,
        correlacao_fundamental=correlacao_fundamental,
        pearson_fundamental=pearson_fundamental,
        Hamiltoniano=H, estado_fundamental=gamma0, Q=np.asarray(Q), q=np.asarray(q),
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
    for epsilon in epsilons:
        problema = DualEntropico(H, Q, q, epsilon)
        learnrate = configuracao["fator_learnrate"] * epsilon
        multiplicadores_iniciais = np.zeros(len(Q)) if u_inicial is None else u_inicial.copy()
        resultado = otimizar_dual(
            problema,
            u_inicial=u_inicial,
            metodo=configuracao["metodo"],
            tolerancia=configuracao["tolerancia"],
            maxiter=configuracao["maxiter"],
            learnrate=learnrate,
        )
        u_inicial = resultado["multiplicadores"]
        gamma = resultado["estado"]
        energia_reconstruida = np.trace(H @ gamma).real
        erro_energia = abs(energia_reconstruida - energia_fundamental)
        entropia = entropia_von_neumann(gamma)
        distancia = np.linalg.norm(gamma - gamma0, ord="fro")
        norma_residuos = resultado["norma_residuos"]
        momentos = momentos_observaveis_locais(gamma, dims, sistemas_posicao, posicoes)
        correlacao_reconstruida = momentos["covariancia"]
        pearson_reconstruido = momentos["pearson"]
        if epsilon == epsilon_visualizacao:
            estado_visualizacao = gamma, pearson_reconstruido
        dados.append((
            energia_reconstruida, erro_energia, entropia,
            distancia, norma_residuos, correlacao_reconstruida, pearson_reconstruido,
        ))
        marginais_reconstruidas = {
            sistema: partial_trace(gamma, keep=sistema, dims=dims)
            for sistema in marginais
        }
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
        densidade = distribuicao_conjunta_posicoes(gamma_visualizacao, grade, sistemas_posicao, dims)

        figura, eixo = plt.subplots(figsize=(6, 5), layout="constrained")
        imagem = eixo.contourf(grade, grade, densidade.T, levels=40, cmap="viridis")
        eixo.set_aspect("equal")
        eixo.set_xlabel(r"$x_0$")
        eixo.set_ylabel(r"$x_1$")
        eixo.set_title(
            f"Distribuição conjunta de posições (ε = {epsilon_visualizacao:g})\n"
            f"Correlação de Pearson = {pearson_visualizacao:.4f}"
        )
        figura.colorbar(imagem, ax=eixo, label=r"$P(x_0, x_1)$")

    plt.show()
