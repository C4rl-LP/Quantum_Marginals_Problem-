"""Testa o dual entrópico com as marginais incompatíveis AB e AC de Bell."""

import argparse
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulos.diagnosticos_reconstrucao import avaliar_testemunha
from modulos.dual_entropico import DualEntropico
from modulos.estados import entropia_von_neumann, ketbra
from modulos.marginais import restricoes_lineares
from modulos.operadores import embed_operator
from modulos.otimizacao import otimizar_dual
from modulos.validacao import comparar_marginais


def executar(epsilon=0.1, maxiter=500):
    """Retorna o certificado de incompatibilidade, a saída dual e os erros locais."""
    dims = (2, 2, 2)
    bell = ketbra(np.array([1, 0, 0, 1]) / np.sqrt(2))
    marginais = {(0, 1): bell, (0, 2): bell}

    # Se ambas as marginais fossem satisfeitas, a média de W seria 2.
    W = embed_operator(bell, (0, 1), dims) + embed_operator(bell, (0, 2), dims)
    alvo = 2 * np.trace(bell @ bell).real
    maximo, separacao = avaliar_testemunha(W, alvo)
    # Na direção tW, o valor dual é pelo menos
    # t * separacao - epsilon * log(dimensão), para t >= 0.

    Q, q = restricoes_lineares(marginais, dims)
    problema = DualEntropico(np.zeros_like(W), Q, q, epsilon, forma="logaritmico")
    resultado = otimizar_dual(
        problema, learnrate=0.2 * problema.epsilon, maxiter=maxiter, tolerancia=1e-8,
    )
    estado = resultado["estado"]
    comparacao = comparar_marginais(estado, {"AB": bell, "AC": bell}, dims)
    return {
        "alvo_testemunha": alvo,
        "maximo_testemunha": maximo,
        "cota_inferior_residuo": separacao / np.sqrt(2),
        "otimizacao": resultado,
        "erros_marginais": {nome: dados["frobenius_error"] for nome, dados in comparacao.items()},
        "marginal_prescrita": bell,
        "marginais_obtidas": {nome: dados["calculated_marginal"] for nome, dados in comparacao.items()},
    }


def gerar_figuras_slides():
    """Grava e retorna os caminhos dos dois PNGs do Beamer."""
    import matplotlib.pyplot as plt
    from tests.qubits.figuras import matriz_densidade, salvar

    dados = executar()
    resultado = dados["otimizacao"]
    historico = np.asarray([
        *resultado["historico"],
        (resultado["valor_dual"], resultado["norma_residuos"]),
    ])
    figura, eixos = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for eixo, valores, rotulo in zip(eixos, historico.T, ("Valor dual", "Norma do resíduo")):
        eixo.plot(valores, color="C0")
        eixo.set(xlabel="Atualizações", ylabel=rotulo)
        eixo.grid(alpha=0.25)
    eixos[1].axhline(
        dados["cota_inferior_residuo"], color="0.4", ls="--", label="Cota da testemunha",
    )
    eixos[1].legend()
    caminho_dual = salvar(figura, "bell_dual_residuo")

    ab = dados["marginais_obtidas"]["AB"]
    ac = dados["marginais_obtidas"]["AC"]
    np.testing.assert_allclose(ab, ac, atol=1e-12)
    figura, eixos = plt.subplots(1, 2, figsize=(6, 2.8), layout="constrained")
    matriz_densidade(eixos[0], dados["marginal_prescrita"], "Bell prescrita")
    matriz_densidade(eixos[1], ab, "Marginal obtida (AB e AC)")
    return caminho_dual, salvar(figura, "bell_marginais")


def main():
    """Mostra o certificado e a evolução do dual e dos resíduos no terminal."""
    epsilon = 0.1
    dados = executar(epsilon)
    resultado = dados["otimizacao"]
    estado = resultado["estado"]
    print("Marginais prescritas: ρ_AB = ρ_AC = |Φ⁺⟩⟨Φ⁺|.")
    print("As duas marginais induzem I₂/2 em A.")
    print(f"Valor exigido da testemunha: {dados['alvo_testemunha']:.6f}")
    print(f"Máximo físico da testemunha: {dados['maximo_testemunha']:.6f}")
    print(f"Cota inferior para a norma do resíduo: {dados['cota_inferior_residuo']:.6f}")
    print("O dual é ilimitado acima na direção da testemunha.")
    print(f"\nDual logarítmico com ε={epsilon:g} e H=0")
    print(f"{'Atualizações':>12} {'Valor dual':>14} {'Norma do resíduo':>18}")
    historico = resultado["historico"]
    indices = np.unique(np.linspace(0, len(historico) - 1, 5, dtype=int))
    for i in indices:
        valor, residuo = historico[i]
        print(f"{i:12d} {valor:14.6f} {residuo:18.6f}")
    print(f"{len(historico):12d} {resultado['valor_dual']:14.6f} {resultado['norma_residuos']:18.6f}")
    print(f"\nConvergiu: {resultado['convergiu']}")
    print(f"Traço: {np.trace(estado).real:.6f} | Entropia: {entropia_von_neumann(estado):.6f} nats")
    for nome, erro in dados["erros_marginais"].items():
        print(f"Erro de Frobenius em {nome}: {erro:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figuras-slides", action="store_true", help="Grava os PNGs do Beamer.")
    if parser.parse_args().figuras_slides:
        for caminho in gerar_figuras_slides():
            print(caminho)
    else:
        main()
