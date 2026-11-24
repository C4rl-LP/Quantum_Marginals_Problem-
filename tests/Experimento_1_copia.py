import argparse
import sys
from pathlib import Path

import numpy as np

# Permite executar este arquivo diretamente a partir de qualquer diretório.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulos.dual_entropico import DualEntropico
from modulos.otimizacao import otimizar_dual
from modulos.operadores import embed_operator, partial_trace
from modulos.Hamiltoniano_ex import Hamiltoniano_ZZ, Hamiltoniano_XY, dimensoes_qubits


PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.diag([1, -1])


def contruindo_problema_dual(rho_local, epsilon, N, H=None):
    dims = dimensoes_qubits(N)
    d = 2**N
    if H is None:
        H = np.zeros((d, d), dtype=complex)

    base_sem_traco = (PAULI_X, PAULI_Y, PAULI_Z)
    M = [np.eye(d, dtype=complex)]
    M.extend(
        embed_operator(base, (i,), dims=dims)
        for i in range(N)
        for base in base_sem_traco
    )
    momentos = [np.trace(rho_local @ base).real for base in base_sem_traco]
    b = np.array([1.0] + momentos * N, dtype=float)
    return DualEntropico(H, M, b, epsilon)


def resolver_cadeia(rho_local, H, epsilon, N, energia_minima):
    problema = contruindo_problema_dual(rho_local, epsilon, N, H)
    # Começa com traço <= 1, mesmo quando H/epsilon tem autovalores grandes.
    u_inicial = np.zeros(len(problema.q))
    u_inicial[0] = energia_minima - epsilon * np.log(H.shape[0])
    return otimizar_dual(
        problema, u_inicial=u_inicial, tolerancia=1e-10,
        maxiter=5000, learnrate=0.2 * epsilon,
    )


def entropia_von_neumann(estado):
    """Calcula a entropia de von Neumann em nats."""
    autovalores = np.linalg.eigvalsh(estado)
    positivos = autovalores[autovalores > 0]
    return float(-np.sum(positivos * np.log(positivos)))


def negatividade(estado):
    """Negatividade entre o primeiro qubit e o restante da cadeia."""
    tamanho = estado.shape[0]
    resto = tamanho // 2
    matriz = np.asarray(estado).reshape(2, resto, 2, resto)
    transposta_parcial = matriz.transpose(2, 1, 0, 3).reshape(tamanho, tamanho)
    autovalores = np.linalg.eigvalsh(transposta_parcial)
    return float(np.abs(autovalores[autovalores < 0]).sum())


def estado_analitico(H, epsilon):
    """Estado de Gibbs; as simetrias globais X e Z impõem marginais I₂/2."""
    valores, vetores = np.linalg.eigh(H)
    pesos = np.exp(-(valores - valores[0]) / epsilon)
    return (vetores * pesos) @ vetores.conj().T / pesos.sum()


def executar_caso(nome, H, epsilon, N, rho_local):
    dims = dimensoes_qubits(N)
    energia_minima = np.linalg.eigvalsh(H)[0]
    resultado = resolver_cadeia(rho_local, H, epsilon, N, energia_minima)
    gamma = resultado["estado"]
    erros_marginais = [
        np.linalg.norm(partial_trace(gamma, keep=(i,), dims=dims) - rho_local, ord="fro")
        for i in range(N)
    ]

    referencia = estado_analitico(H, epsilon)
    erro = float(np.linalg.norm(gamma - referencia, ord="fro"))
    residuos = float(np.linalg.norm(resultado["residuos"]))
    pureza = float(np.trace(gamma @ gamma).real)
    entropia = entropia_von_neumann(gamma)
    neg = negatividade(gamma)
    print(
        f"{nome:16s} eps={epsilon:.3g}  erro={erro:.3e}  "
        f"resíduo={residuos:.3e}  pureza={pureza:.6f}  "
        f"entropia={entropia:.6f} nats  negatividade={neg:.6f}\n"
        f"  Maior erro entre as {N} marginais={max(erros_marginais):.3e}"
    )
    if not resultado["convergiu"] or residuos > 1e-6 or erro > 1e-5:
        print("  Aviso: confira a convergência e os resíduos do otimizador.")
    print("  Estado reconstruído (base computacional):")
    print(np.array2string(gamma, precision=4, suppress_small=True))
    return gamma


def varrer_epsilon(valores_epsilon=None, J=1.0, N=3):
    """Retorna a figura e as medidas dos três casos para marginais I₂/2."""
    import matplotlib.pyplot as plt

    dimensoes_qubits(N)
    if valores_epsilon is None:
        valores_epsilon = np.linspace(0.1, 2.0, 41)
    valores_epsilon = np.asarray(valores_epsilon, dtype=float)
    if (
        valores_epsilon.ndim != 1
        or valores_epsilon.size == 0
        or not np.isfinite(valores_epsilon).all()
        or np.any(valores_epsilon <= 0)
    ):
        raise ValueError("valores_epsilon deve ser um vetor não vazio de valores positivos e finitos.")
    valores_epsilon = np.sort(valores_epsilon)

    rho_local = np.eye(2, dtype=complex) / 2
    casos = (
        ("Sem interação", None),
        ("Ising ZZ", Hamiltoniano_ZZ),
        ("Interação XY", Hamiltoniano_XY),
    )
    dados = {"epsilon": valores_epsilon}
    for nome, construir_H in casos:
        entropias, energias, negatividades = [], [], []
        H = np.zeros((2**N, 2**N)) if construir_H is None else construir_H(J, N)
        energia_minima = np.linalg.eigvalsh(H)[0]
        for epsilon in valores_epsilon:
            resultado = resolver_cadeia(rho_local, H, epsilon, N, energia_minima)
            if not resultado["convergiu"]:
                raise RuntimeError(
                    f"{nome}, ε={epsilon:g}: otimização não convergiu "
                    f"(resíduo={resultado['norma_residuos']:.3e})."
                )
            gamma = resultado["estado"]
            entropias.append(entropia_von_neumann(gamma))
            energias.append(float(np.trace(gamma @ H).real))
            negatividades.append(negatividade(gamma))
        dados[nome] = {
            "entropia": np.asarray(entropias),
            "energia": np.asarray(energias),
            "negatividade": np.asarray(negatividades),
        }

    figura, eixos = plt.subplots(3, 1, sharex=True, figsize=(8, 8), layout="constrained")
    figura.suptitle(f"Experimento 1 — N = {N}, J = {J:g}")
    medidas = (
        ("entropia", "Entropia (nats)"),
        ("energia", "Energia média ⟨H⟩"),
        ("negatividade", "Negatividade (1 | restante)"),
    )
    for eixo, (medida, rotulo) in zip(eixos, medidas):
        for (nome, _), estilo in zip(casos, ("-", "--", "-.")):
            eixo.plot(valores_epsilon, dados[nome][medida], linestyle=estilo, label=nome)
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.25)
    eixos[0].legend()
    eixos[-1].set_xlabel("Regularização ε")
    return figura, dados


def main(N=3):
    dimensoes_qubits(N)
    J = 1.0
    epsilon = 0.5
    rho_local = np.eye(2, dtype=complex) / 2
    H_sem_interacao = np.zeros((2**N, 2**N), dtype=complex)
    casos = (
        ("sem interação", H_sem_interacao),
        ("Ising ZZ", Hamiltoniano_ZZ(J, N)),
        ("Interação XY", Hamiltoniano_XY(J, N)),
    )

    print(f"Parâmetros: N={N}, J={J:g}, epsilon={epsilon:g}; rho_i=I_2/2\n")
    for nome, H in casos:
        executar_caso(nome, H, epsilon, N, rho_local)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=3, help="Número de qubits da cadeia.")
    parser.add_argument(
        "--varrer-epsilon", type=Path, metavar="ARQUIVO",
        help="Salva o gráfico da varredura em ε.",
    )
    args = parser.parse_args()
    if args.qubits < 2:
        parser.error("--qubits deve ser maior ou igual a 2.")
    if args.varrer_epsilon is None:
        main(args.qubits)
    else:
        figura, _ = varrer_epsilon(N=args.qubits)
        figura.savefig(args.varrer_epsilon, dpi=180)
