import argparse
import sys
from pathlib import Path

import numpy as np

# Permite executar este arquivo diretamente a partir de qualquer diretório.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulos.estados import entropia_von_neumann, negatividade
from modulos.dual_entropico import DualEntropico
from modulos.otimizacao import otimizar_dual
from modulos.operadores import partial_trace

# H_ZZ = -J sigma_z \otimes sigma_z
# H_XY = -J (sigma_x \otimes sigma_x + sigma_y \otimes sigma_y)
def Hamiltoniano_ZZ(J):
    sigma_z = np.diag([1, -1])
    return -J * np.kron(sigma_z, sigma_z)

def Hamiltoniano_XY(J):
    sigma_x = np.array([[0,1],
                        [1,0]])
    sigma_y = np.array([[0,-1j],
                        [1j,0]], dtype=complex)
    return -J * (np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y))

def contruindo_problema_dual(p_A, p_B, epsilon, H=None):

    if H is None:
        H = np.zeros((4, 4), dtype=complex)
    sigma_z = np.diag([1,-1])
    sigma_x = np.array([[0,1],
                        [1,0]])
    sigma_y = np.array([[0,-1j],
                        [1j,0]], dtype=complex)

    base_sem_traco = (sigma_x, sigma_y, sigma_z)
    M = [np.eye(4, dtype=complex)]
    M.extend(np.kron(base, np.eye(2, dtype=complex)) for base in base_sem_traco)
    M.extend(np.kron(np.eye(2, dtype=complex), base) for base in base_sem_traco)

    b = np.array(
        [1.0]
        + [np.trace(p_A @ base).real for base in base_sem_traco]
        + [np.trace(p_B @ base).real for base in base_sem_traco],
        dtype=float,
    )

    return DualEntropico(H, M, b, epsilon)




def estado_analitico(caso, epsilon, J):
    """Estados de referência para as três escolhas de Hamiltoniano."""
    identidade = np.eye(4, dtype=complex)
    if caso == "sem interação":
        return identidade / 4
    if caso == "Ising ZZ":
        zz = np.kron(np.diag([1, -1]), np.diag([1, -1]))
        return (identidade + np.tanh(J / epsilon) * zz) / 4
    H = Hamiltoniano_XY(J)
    valores, vetores = np.linalg.eigh(-H / epsilon)
    pesos = np.exp(valores - np.max(valores))
    return (vetores * pesos) @ vetores.conj().T / pesos.sum()


def executar_caso(nome, H, epsilon, J, rho_local):
    problema = contruindo_problema_dual(rho_local, rho_local, epsilon, H)
    resultado = otimizar_dual(problema, tolerancia=1e-10, maxiter=5000)
    gamma = resultado["estado"]
    dims = (2, 2)
    marginal_A = partial_trace(gamma, keep=(0,), dims=dims)
    marginal_B = partial_trace(gamma, keep=(1,), dims=dims)
    erro_marginal_A = float(np.linalg.norm(marginal_A - rho_local, ord="fro"))
    erro_marginal_B = float(np.linalg.norm(marginal_B - rho_local, ord="fro"))

    referencia = estado_analitico(nome, epsilon, J)
    erro = float(np.linalg.norm(gamma - referencia, ord="fro"))
    residuos = float(np.linalg.norm(resultado["residuos"]))
    pureza = float(np.trace(gamma @ gamma).real)
    entropia = entropia_von_neumann(gamma)
    neg = negatividade(gamma)
    print(
        f"{nome:16s} eps={epsilon:.3g}  erro={erro:.3e}  "
        f"resíduo={residuos:.3e}  pureza={pureza:.6f}  "
        f"entropia={entropia:.6f} nats  negatividade={neg:.6f}\n"
        f"  Erro marginal A={erro_marginal_A:.3e}  Erro marginal B={erro_marginal_B:.3e}"
    )
    if not resultado["convergiu"] or residuos > 1e-6 or erro > 1e-5:
        print("  Aviso: confira a convergência e os resíduos do otimizador.")
    print("  Estado reconstruído (base |00>, |01>, |10>, |11>):")
    print(np.array2string(gamma, precision=4, suppress_small=True))
    return gamma


def varrer_epsilon(valores_epsilon=None, J=1.0):
    """Retorna a figura e as medidas dos três casos para marginais I₂/2."""
    import matplotlib.pyplot as plt

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
        u_inicial = None
        H = np.zeros((4, 4)) if construir_H is None else construir_H(J)
        for epsilon in valores_epsilon[::-1]:
            problema = contruindo_problema_dual(rho_local, rho_local, epsilon, H)
            resultado = otimizar_dual(
                problema, u_inicial=u_inicial, tolerancia=1e-10, maxiter=5000
            )
            if not resultado["convergiu"]:
                raise RuntimeError(
                    f"{nome}, ε={epsilon:g}: otimização não convergiu "
                    f"(resíduo={resultado['norma_residuos']:.3e})."
                )
            u_inicial = resultado["multiplicadores"]
            gamma = resultado["estado"]
            entropias.append(entropia_von_neumann(gamma))
            energias.append(float(np.trace(gamma @ H).real))
            negatividades.append(negatividade(gamma))
        dados[nome] = {
            "entropia": np.asarray(entropias[::-1]),
            "energia": np.asarray(energias[::-1]),
            "negatividade": np.asarray(negatividades[::-1]),
        }

    figura, eixos = plt.subplots(3, 1, sharex=True, figsize=(8, 8), layout="constrained")
    figura.suptitle(f"Experimento 1 — J = {J:g}")
    medidas = (
        ("entropia", "Entropia (nats)"),
        ("energia", "Energia média ⟨H⟩"),
        ("negatividade", "Negatividade"),
    )
    for eixo, (medida, rotulo) in zip(eixos, medidas):
        for (nome, _), estilo in zip(casos, ("-", "--", "-.")):
            eixo.plot(valores_epsilon, dados[nome][medida], linestyle=estilo, label=nome)
        eixo.set_ylabel(rotulo)
        eixo.grid(alpha=0.25)
    eixos[0].legend()
    eixos[-1].set_xlabel("Regularização ε")
    return figura, dados


def main():

    J = 1.0
    epsilon = 0.5
    rho_local = np.eye(2, dtype=complex) / 2
    H_sem_interacao = np.zeros((4, 4), dtype=complex)
    casos = (
        ("sem interação", H_sem_interacao),
        ("Ising ZZ", Hamiltoniano_ZZ(J)),
        ("Interação XY", Hamiltoniano_XY(J)),
    )

    print(f"Parâmetros: J={J:g}, epsilon={epsilon:g}; rho_A=rho_B=I_2/2\n")
    for nome, H in casos:
        executar_caso(nome, H, epsilon, J, rho_local)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--varrer-epsilon", type=Path, metavar="ARQUIVO",
        help="Salva o gráfico da varredura em ε.",
    )
    args = parser.parse_args()
    if args.varrer_epsilon is None:
        main()
    else:
        figura, _ = varrer_epsilon()
        figura.savefig(args.varrer_epsilon, dpi=180)
