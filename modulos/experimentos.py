"""Geradores aleatórios e resumos para experimentos de compatibilidade."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .estados import ketbra
from .operadores import partial_trace


def random_density_matrix(d: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Gera uma matriz densidade aleatória de dimensão d usando o método de Ginibre.

    A construção é:
        G = matriz complexa d×d de entradas i.i.d. N(0,1) + i N(0,1)
        rho = G G† / Tr(G G†)

    Garantias numéricas verificadas:
        - rho = rho†   (hermiticidade)
        - rho >= 0     (positividade semidefinida)
        - Tr(rho) = 1  (normalização)

    Parameters
    ----------
    d : int
        Dimensão do espaço de Hilbert (d=2 para qubit, d=4 para 2 qubits).
    rng : np.random.Generator, opcional
        Gerador de números aleatórios (para reprodutibilidade).

    Returns
    -------
    rho : np.ndarray, shape (d, d), dtype complex128
        Matriz densidade válida.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Gera matriz complexa aleatória
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))

    # Constrói rho = G G†
    rho = G @ G.conj().T

    # Normaliza pelo traço
    rho = rho / np.trace(rho)

    # Garante hermiticidade perfeita numericamente (elimina erros de arredondamento)
    rho = (rho + rho.conj().T) / 2.0

    return rho


def build_product_marginals(
    rho_A: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrói rho_AB = rho_A ⊗ sigma_B e rho_AC = rho_A ⊗ sigma_C,
    onde sigma_B e sigma_C são matrizes densidade aleatórias de dimensão 2.

    Garante: Tr_B(rho_AB) = rho_A  e  Tr_C(rho_AC) = rho_A.
    Note: admite sempre extensão global trivial rho_ABC = rho_A ⊗ sigma_B ⊗ sigma_C.

    Parameters
    ----------
    rho_A : ndarray (2, 2)
        Marginal prescrita para o subsistema A.
    rng : np.random.Generator, opcional

    Returns
    -------
    rho_AB : ndarray (4, 4)
    rho_AC : ndarray (4, 4)
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_B = random_density_matrix(2, rng=rng)
    sigma_C = random_density_matrix(2, rng=rng)

    rho_AB = np.kron(rho_A, sigma_B)
    rho_AC = np.kron(rho_A, sigma_C)

    return rho_AB, rho_AC


def build_correlated_marginals(
    rho_A: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    num_kraus: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrói rho_AB e rho_AC correlacionados que mantêm rho_A como marginal.

    Estratégia (purificação aleatória via decomposição espectral):
    1. Decompõe rho_A = sum_k lambda_k |k><k|
    2. Constrói purificação em AB: |Psi_AB> = sum_k sqrt(lambda_k) |k>_A ⊗ |phi_k>_B
       onde |phi_k> são vetores aleatórios ortonormalizados.
    3. O estado puro |Psi_AB><Psi_AB| tem Tr_B(.) = rho_A por construção.
    4. Aplica canal quântico aleatório para obter estado misto correlacionado.

    Parameters
    ----------
    rho_A : ndarray (2, 2)
        Marginal prescrita para o subsistema A.
    rng : np.random.Generator, opcional
    num_kraus : int
        Número de operadores de Kraus para o canal aleatório (controla
        o grau de mistura introduzido).

    Returns
    -------
    rho_AB : ndarray (4, 4)  com Tr_B(rho_AB) ≈ rho_A
    rho_AC : ndarray (4, 4)  com Tr_C(rho_AC) ≈ rho_A
    """
    if rng is None:
        rng = np.random.default_rng()

    def _purification_state(rho: np.ndarray) -> np.ndarray:
        """
        Constrói uma purificação de rho em espaço ampliado d×d.
        Retorna a matriz densidade do estado puro (d^2 × d^2 não, d × d do estado puro
        no espaço AB de dimensão d×d = 4×4 para qubit).
        """
        d = rho.shape[0]
        # Decomposição espectral
        evals, evecs = np.linalg.eigh(rho)
        # Garante não-negatividade (erros numéricos)
        evals = np.maximum(evals, 0.0)

        # Vetores ortonormais aleatórios para o ambiente
        # (aqui o ambiente tem a mesma dimensão que o sistema)
        U_env, _ = np.linalg.qr(
            rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        )

        # |Psi> = sum_k sqrt(lambda_k) |k>_sys ⊗ U_env|k>_env
        psi = np.zeros(d * d, dtype=complex)
        for k in range(d):
            sys_vec = evecs[:, k]           # autovetor do sistema
            env_vec = U_env[:, k]           # vetor ortonormal do ambiente
            psi += np.sqrt(evals[k]) * np.kron(sys_vec, env_vec)

        # Normaliza (deve já ser 1, mas garante numericamente)
        norm = np.linalg.norm(psi)
        if norm > 1e-14:
            psi = psi / norm

        return ketbra(psi)

    def _apply_random_channel_on_B(rho_AB_pure: np.ndarray) -> np.ndarray:
        """
        Aplica um canal quântico aleatório no subsistema B de rho_AB.
        O canal é descrito por operadores de Kraus aleatórios normalizados.
        Preserva Tr_B(rho_AB) = rho_A.
        """
        d_A, d_B = 2, 2
        d_AB = d_A * d_B

        # Gera operadores de Kraus aleatórios para o canal em B
        # K_i: d_B × d_B tal que sum_i K_i† K_i = I_B
        # Método: gera uma isometria d_B × (d_B * num_kraus) e extrai blocos
        d_out = d_B * num_kraus
        V, _ = np.linalg.qr(
            rng.standard_normal((d_out, d_out)) + 1j * rng.standard_normal((d_out, d_out))
        )
        # Toma as primeiras d_B colunas como isometria d_out × d_B
        V_iso = V[:d_B * num_kraus, :d_B]  # shape (num_kraus * d_B, d_B)

        kraus_ops = [V_iso[i * d_B:(i + 1) * d_B, :] for i in range(num_kraus)]

        # Verifica completeza: sum K_i† K_i = I
        completeness = sum(K.conj().T @ K for K in kraus_ops)
        # Re-normaliza se necessário
        if not np.allclose(completeness, np.eye(d_B), atol=1e-10):
            # Usa SVD para obter operadores de Kraus válidos
            U_q, _, _ = np.linalg.svd(completeness)
            kraus_ops = [K @ U_q.conj().T for K in kraus_ops]
            # Normaliza
            completeness2 = sum(K.conj().T @ K for K in kraus_ops)
            scale = np.sqrt(np.linalg.norm(completeness2, 2))
            kraus_ops = [K / scale for K in kraus_ops]

        # Aplica o canal: rho_AB -> sum_k (I_A ⊗ K_k) rho_AB (I_A ⊗ K_k)†
        rho_out = np.zeros((d_AB, d_AB), dtype=complex)
        for K in kraus_ops:
            # I_A ⊗ K_k
            op = np.kron(np.eye(d_A, dtype=complex), K)
            rho_out += op @ rho_AB_pure @ op.conj().T

        # Renormaliza (preserva Tr)
        tr_out = np.trace(rho_out)
        if abs(tr_out) > 1e-14:
            rho_out = rho_out / tr_out

        # Garante hermiticidade
        rho_out = (rho_out + rho_out.conj().T) / 2.0

        return rho_out

    # Constrói purificação de rho_A em espaço AB
    rho_AB_pure = _purification_state(rho_A)
    # Aplica canal aleatório em B para introduzir correlações diferentes
    rho_AB = _apply_random_channel_on_B(rho_AB_pure)

    # Repete de forma independente para AC
    rho_AC_pure = _purification_state(rho_A)
    rho_AC = _apply_random_channel_on_B(rho_AC_pure)  # canal aleatório independente em C

    return rho_AB, rho_AC


def check_overlap_consistency(
    rho_AB: np.ndarray,
    rho_AC: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[float, bool]:
    """
    Calcula o erro de sobreposição ||Tr_B(rho_AB) - Tr_C(rho_AC)||_F
    e determina se a condição de sobreposição é satisfeita.

    Reutiliza partial_trace do projeto com a assinatura correta:
        partial_trace(rho, keep=[...], dims=[...])

    Parameters
    ----------
    rho_AB : ndarray (4, 4)  — estado bipartido AB
    rho_AC : ndarray (4, 4)  — estado bipartido AC
    tol : float
        Tolerância numérica para compatibilidade.

    Returns
    -------
    delta : float
        Norma de Frobenius da diferença das marginais em A.
    compatible : bool
        True se delta < tol.
    """
    # Tr_B(rho_AB): mantém subsistema 0 (A), traça 1 (B)  — dims=[2,2]
    rho_A_from_AB = partial_trace(rho_AB, keep=[0], dims=[2, 2])
    # Tr_C(rho_AC): mantém subsistema 0 (A), traça 1 (C)  — dims=[2,2]
    rho_A_from_AC = partial_trace(rho_AC, keep=[0], dims=[2, 2])

    delta = float(np.linalg.norm(rho_A_from_AB - rho_A_from_AC, 'fro'))
    compatible = delta < tol

    return delta, compatible


def print_experiment_stats(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """
    Imprime estatísticas do DataFrame e retorna as métricas em um dicionário.
    """
    n_total    = len(df)
    n_overlap  = int(df["overlap_compatible"].sum())
    n_feasible = int(df["is_feasible"].sum())

    # Pares que satisfazem overlap mas são globalmente incompatíveis
    mask_overlap_infeasible = df["overlap_compatible"] & ~df["is_feasible"]
    n_overlap_infeasible    = int(mask_overlap_infeasible.sum())

    # Pares que satisfazem overlap e são globalmente compatíveis
    mask_overlap_feasible   = df["overlap_compatible"] & df["is_feasible"]
    n_overlap_feasible      = int(mask_overlap_feasible.sum())

    # P(globalmente compatível | overlap compatível)
    p_cond = n_overlap_feasible / n_overlap if n_overlap > 0 else float("nan")

    print(f"\n{'═'*65}")
    print(f"  Experimento: {label}")
    print(f"{'═'*65}")
    print(f"  Total de amostras:                        {n_total:6d}")
    print(f"  Overlap compatível:                       {n_overlap:6d}  ({100*n_overlap/n_total:.1f}%)")
    print(f"  Globalmente factível:                     {n_feasible:6d}  ({100*n_feasible/n_total:.1f}%)")
    print(f"  Overlap ✓  e  SDP infactível:             {n_overlap_infeasible:6d}")
    print(f"  Overlap ✓  e  SDP factível:               {n_overlap_feasible:6d}")
    if n_overlap > 0:
        print(f"  P(factível | overlap ✓):                  {p_cond:.4f}")
    else:
        print(f"  P(factível | overlap ✓):                  N/A (sem amostras com overlap)")
    print(f"  delta_overlap — média:                    {df['delta_overlap'].mean():.4e}")
    print(f"  delta_overlap — mediana:                  {df['delta_overlap'].median():.4e}")
    print(f"  solver_time   — média (s):                {df['solver_time'].mean():.4f}")

    return {
        "label":                label,
        "n_total":              n_total,
        "n_overlap":            n_overlap,
        "n_feasible":           n_feasible,
        "n_overlap_infeasible": n_overlap_infeasible,
        "n_overlap_feasible":   n_overlap_feasible,
        "p_cond_feasible_given_overlap": p_cond,
    }
