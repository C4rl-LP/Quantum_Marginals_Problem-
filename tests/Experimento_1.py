import numpy as np
import sys
from pathlib import Path

# Permite executar este arquivo diretamente a partir de qualquer diretório.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulos.dual_entropico import DualEntropico
from modulos.otimizacao import otimizar_dual


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
    """Monta o dual para marginais de dois qubits, com traço global um."""
    if H is None:
        H = np.zeros((4, 4), dtype=complex)
    sigma_z = np.diag([1,-1])
    sigma_x = np.array([[0,1],
                        [1,0]])
    sigma_y = np.array([[0,-1j],
                        [1j,0]], dtype=complex)
    # A identidade global fixa o traço; as três Pauli fixam cada marginal.
    # Usar somente operadores sem traço evita repetir a restrição do traço.
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


def negatividade(estado):
    """Negatividade do estado bipartido, transpondo o segundo qubit."""
    matriz = np.asarray(estado).reshape(2, 2, 2, 2)
    transposta_parcial = matriz.transpose(0, 3, 2, 1).reshape(4, 4)
    return float((np.abs(np.linalg.eigvalsh(transposta_parcial)).sum() - 1) / 2)


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
    gamma = (gamma + gamma.conj().T) / 2
    gamma /= np.trace(gamma).real
    referencia = estado_analitico(nome, epsilon, J)
    erro = float(np.linalg.norm(gamma - referencia, ord="fro"))
    residuos = float(np.linalg.norm(resultado["residuos"]))
    pureza = float(np.trace(gamma @ gamma).real)
    neg = negatividade(gamma)
    print(
        f"{nome:16s} eps={epsilon:.3g}  erro={erro:.3e}  "
        f"resíduo={residuos:.3e}  pureza={pureza:.6f}  negatividade={neg:.6f}"
    )
    if not resultado["convergiu"] or residuos > 1e-6 or erro > 1e-5:
        print("  Aviso: confira a convergência e os resíduos do otimizador.")
    print("  Estado reconstruído (base |00>, |01>, |10>, |11>):")
    print(np.array2string(gamma, precision=4, suppress_small=True))
    return gamma


def main():
    # Mesmas marginais maximamente mistas em todos os casos.
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

    print("No XY, epsilon menor favorece o estado fundamental emaranhado |Psi+>.")


if __name__ == "__main__":
    main()
