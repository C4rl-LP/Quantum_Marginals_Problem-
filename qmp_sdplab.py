"""
Quantum Marginal Problem (QMP) via Semidefinite Programming (SDP) using SDPLab.

Este módulo implementa a resolução numérica do problema de representabilidade mista
de marginais quânticas (Quantum Marginal Problem) utilizando a biblioteca SDPLab (`sdplab`)
e o seu backend de resolução cônica `run_cvxpy_solver`.

Regimes analisados:
1. GHZ: mixed = sim, pure = sim.
2. Maximamente misto: mixed = sim, pure = não.
3. Incompatibilidade (Bell em AB e AC): mixed = não, pure = não.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Verificação e importação da API do SDPLab e SpaceCore
# ---------------------------------------------------------------------------
try:
    import spacecore
    from spacecore import Context, DenseVectorSpace, HermitianSpace, NumpyOps
    import sdplab
    from sdplab import DenseConstraintOp, SDPProblem
    from sdplab.solvers import run_cvxpy_solver
except ImportError as err:
    raise ImportError(
        f"Erro ao importar sdplab ou spacecore: {err}.\n"
        "Certifique-se de que sdplab e spacecore estão instalados corretamente "
        "(recomenda-se spacecore==0.4.2 para compatibilidade com sdplab==0.0.1)."
    ) from err


# ===========================================================================
# 1. Definição de estados e operadores quânticos elementares
# ===========================================================================

def ketbra(psi: np.ndarray) -> np.ndarray:
    """
    Constrói o projetor |psi><psi| para um vetor de estado quântico |psi>.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return np.outer(psi, psi.conj())


def tensor(*args: np.ndarray) -> np.ndarray:
    """
    Calcula o produto de Kronecker (tensorial) sequencial de múltiplos operadores/estados.
    """
    res = args[0]
    for op in args[1:]:
        res = np.kron(res, op)
    return res


# Qubits computacionais
zero = np.array([1.0, 0.0], dtype=complex)
one = np.array([0.0, 1.0], dtype=complex)

# Matrizes de Pauli e Identidade
sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def ghz_state(theta: float = 0.0) -> np.ndarray:
    """
    Gera o estado puro |GHZ_theta> = (|000> + exp(i theta)|111>) / sqrt(2).
    """
    return (tensor(zero, zero, zero) + np.exp(1j * theta) * tensor(one, one, one)) / np.sqrt(2.0)


def bell_phi_plus() -> np.ndarray:
    """
    Gera o estado de Bell puro |Phi+> = (|00> + |11>) / sqrt(2).
    """
    return (tensor(zero, zero) + tensor(one, one)) / np.sqrt(2.0)


# ===========================================================================
# 2. Operadores de traço parcial e manipulação tensorial
# ===========================================================================

def partial_trace(
    rho: np.ndarray,
    keep: Optional[Sequence[int]] = None,
    trace_out: Optional[Sequence[int]] = None,
    dims: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""
    Calcula o traço parcial de uma matriz densidade de forma geral e tensorial.

    A matriz `rho` de dimensão :math:`\prod_k d_k \times \prod_k d_k` é tratada como
    um tensor de ordem :math:`2N`: :math:`\rho_{a_0\dots a_{N-1}, b_0\dots b_{N-1}}`.
    Para os subsistemas eliminados :math:`c \in S^c`, impõe-se a contração
    :math:`a_c = b_c` e soma-se sobre este índice via `np.einsum`.

    Parâmetros:
    -----------
    rho : np.ndarray
        Matriz quadrada do sistema global.
    keep : Sequence[int], opcional
        Índices dos subsistemas mantidos na marginal.
    trace_out : Sequence[int], opcional
        Índices dos subsistemas a serem traçados fora (complemento).
    dims : Sequence[int], opcional
        Dimensões de cada subsistema. Padrão: qubits (dim = 2).

    Retorno:
    --------
    np.ndarray
        Matriz densidade reduzida (marginal) nos subsistemas preservados.
    """
    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        num_qubits = int(round(np.log2(rho.shape[0])))
        dims = [2] * num_qubits
    dims = tuple(dims)
    N = len(dims)

    if keep is not None and trace_out is not None:
        raise ValueError("Especifique apenas `keep` ou `trace_out`, não ambos.")
    if keep is not None:
        keep_systems = tuple(sorted(keep))
    elif trace_out is not None:
        trace_set = set(trace_out)
        keep_systems = tuple(sorted(i for i in range(N) if i not in trace_set))
    else:
        raise ValueError("É necessário especificar `keep` ou `trace_out`.")

    complement = tuple(i for i in range(N) if i not in keep_systems)

    # Tensor de ordem 2N: N índices bra e N índices ket
    rho_tensor = rho.reshape(dims + dims)

    bra_indices = list(range(N))
    ket_indices = list(range(N, 2 * N))

    # Para os subsistemas traçados fora, os índices bra e ket colapsam no mesmo índice mudo
    for c in complement:
        ket_indices[c] = bra_indices[c]

    out_bra = [bra_indices[s] for s in keep_systems]
    out_ket = [ket_indices[s] for s in keep_systems]

    # Contração tensorial direta
    reduced_tensor = np.einsum(rho_tensor, bra_indices + ket_indices, out_bra + out_ket)
    d_out = int(np.prod([dims[s] for s in keep_systems]))
    return reduced_tensor.reshape(d_out, d_out)


def hermitian_basis(d: int) -> List[np.ndarray]:
    r"""
    Gera uma base ortonormal para o espaço vetorial real das matrizes Hermitianas
    :math:`\mathrm{Herm}(d)`, com respeito ao produto interno de Hilbert-Schmidt:
    :math:`\operatorname{Tr}(E_k E_l) = \delta_{kl}`.

    Contém:
    - :math:`d` matrizes diagonais :math:`|i\rangle\langle i|`;
    - :math:`d(d-1)/2` partes reais simétricas :math:`\frac{|i\rangle\langle j| + |j\rangle\langle i|}{\sqrt{2}}`;
    - :math:`d(d-1)/2` partes imaginárias antissimétricas :math:`\frac{i(|i\rangle\langle j| - |j\rangle\langle i|)}{\sqrt{2}}`.
    Total: :math:`d^2` matrizes Hermitianas ortonormais.
    """
    basis: List[np.ndarray] = []
    # Elementos diagonais
    for i in range(d):
        E = np.zeros((d, d), dtype=complex)
        E[i, i] = 1.0
        basis.append(E)

    # Elementos fora da diagonal
    for i in range(d):
        for j in range(i + 1, d):
            # Parte real
            E_re = np.zeros((d, d), dtype=complex)
            E_re[i, j] = 1.0 / np.sqrt(2.0)
            E_re[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(E_re)

            # Parte imaginária
            E_im = np.zeros((d, d), dtype=complex)
            E_im[i, j] = 1.0j / np.sqrt(2.0)
            E_im[j, i] = -1.0j / np.sqrt(2.0)
            basis.append(E_im)

    return basis


def embed_operator(
    O_S: np.ndarray,
    systems: Sequence[int],
    dims: Sequence[int],
) -> np.ndarray:
    r"""
    Incorpora um operador local :math:`O_S` atuando no subsistema :math:`S` no espaço
    global :math:`\mathcal{H} = \bigotimes_k \mathcal{H}_k`, inserindo a identidade
    :math:`I_{d_c}` nos subsistemas do complemento :math:`S^c`.

    Esta operação é fundamental para transcrever a restrição de traço parcial para
    o produto interno global por dualidade:
    :math:`\operatorname{Tr}[ (O_S \otimes I_{S^c}) \rho ] = \operatorname{Tr}[ O_S \operatorname{Tr}_{S^c}(\rho) ]`.
    """
    dims = tuple(dims)
    N = len(dims)
    systems = tuple(systems)
    k = len(systems)
    complement = tuple(i for i in range(N) if i not in systems)

    s_dims = tuple(dims[i] for i in systems)
    O_tensor = O_S.reshape(s_dims + s_dims)

    # Produto externo com identidade para cada sistema complementar
    res = O_tensor
    for c in complement:
        I_c = np.eye(dims[c], dtype=O_S.dtype)
        res = np.multiply.outer(res, I_c)

    # Mapeamento dos eixos originais para a ordem global (a_0,...,a_{N-1}, b_0,...,b_{N-1})
    axis_map_bra = {s: idx for idx, s in enumerate(systems)}
    axis_map_ket = {s: k + idx for idx, s in enumerate(systems)}
    for idx, c in enumerate(complement):
        axis_map_bra[c] = 2 * k + 2 * idx
        axis_map_ket[c] = 2 * k + 2 * idx + 1

    perm = [axis_map_bra[i] for i in range(N)] + [axis_map_ket[i] for i in range(N)]
    res = np.transpose(res, perm)
    d_total = int(np.prod(dims))
    return res.reshape(d_total, d_total)


def parse_subsystem_key(key: Union[str, Sequence[int]], num_qubits: int = 3) -> Tuple[int, ...]:
    """
    Normaliza identificadores de subsistemas como 'AB', 'AC', 'BC', 'A' ou (0, 1), (0, 2).
    """
    if isinstance(key, (tuple, list)):
        return tuple(int(x) for x in key)
    if isinstance(key, str):
        char_map = {chr(ord('A') + i): i for i in range(26)}
        key_upper = key.upper().strip()
        return tuple(char_map[ch] for ch in key_upper if ch in char_map)
    raise TypeError(f"Formato de subsistema inválido: {key!r}")


# ===========================================================================
# 3. Construção das restrições e do problema SDP via SDPLab
# ===========================================================================

def build_qmp_sdp(
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    include_trace_norm: bool = True,
    ctx: Optional[Context] = None,
) -> Tuple[SDPProblem, np.ndarray, np.ndarray]:
    r"""
    Constrói a formulação de Programação Semidefinida (SDP) para o problema de
    representabilidade de marginais quânticas no formato nativo da biblioteca SDPLab.

    Parâmetros:
    -----------
    marginals_dict : Dict[str | tuple, np.ndarray]
        Dicionário associando o identificador do subsistema (ex: 'AB', 'AC', (0, 1))
        à matriz densidade prescrita :math:`\omega_S`.
    dims : Sequence[int], opcional
        Dimensões locais de cada subsistema. Padrão: qubits (2 para cada).
    include_trace_norm : bool, padrão True
        Se True, inclui explicitamente a restrição escalar de normalização Tr(rho) = 1.
    ctx : Context, opcional
        Contexto backend do SpaceCore/SDPLab. Se None, utiliza NumPy complex128.

    Retorno:
    --------
    sdp : SDPProblem
        Instância do problema cônico padrão min <C, X> s.t. A X = b, X >= 0.
    M_full : np.ndarray
        Array (m, d, d) com os observáveis Hermitianos globais de cada restrição.
    b_full : np.ndarray
        Vetor (m,) com os valores reais esperados no lado direito.
    """
    # 1. Normalizar as marginais
    normalized_marginals: Dict[Tuple[int, ...], np.ndarray] = {}
    for k, v in marginals_dict.items():
        sub_tuple = parse_subsystem_key(k)
        mat = np.asarray(v, dtype=complex)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"A marginal {k} deve ser uma matriz quadrada; formato {mat.shape}.")
        normalized_marginals[sub_tuple] = mat

    if dims is None:
        max_idx = max(max(sub) for sub in normalized_marginals.keys())
        dims = [2] * (max_idx + 1)
    dims = tuple(dims)
    d_total = int(np.prod(dims))

    M_list: List[np.ndarray] = []
    b_list: List[float] = []

    # 2. Restrição de normalização: Tr(rho) = Tr(I_d rho) = 1.0
    if include_trace_norm:
        M_list.append(np.eye(d_total, dtype=complex))
        b_list.append(1.0)

    # 3. Restrições afins de traço parcial
    # Para cada subsistema S e cada elemento da base hermitiana E_k em Herm(d_S):
    # Tr[ (E_k \otimes I_{S^c}) \rho ] = Tr[ E_k \omega_S ]
    for systems, omega_S in normalized_marginals.items():
        d_S = omega_S.shape[0]
        expected_d_S = int(np.prod([dims[s] for s in systems]))
        if d_S != expected_d_S:
            raise ValueError(
                f"Dimensão da marginal para o subsistema {systems} é {d_S}, "
                f"mas esperava-se {expected_d_S} conforme dims={dims}."
            )

        basis_S = hermitian_basis(d_S)
        for E in basis_S:
            val = float(np.real(np.trace(E @ omega_S)))
            M_global = embed_operator(E, systems, dims)
            M_list.append(M_global)
            b_list.append(val)

    M_full = np.stack(M_list, axis=0)  # Formato (m, d_total, d_total)
    b_full = np.array(b_list, dtype=float)

    # 4. Construção no SDPLab
    if ctx is None:
        ctx = Context(NumpyOps(), dtype="complex128", check_level="none")

    dom = HermitianSpace(d_total, ctx=ctx)
    cod = DenseVectorSpace((len(b_full),), ctx=ctx)

    # O SDPLab utiliza o pareamento Frobenius: (A X)_i = sum_{pq} T_{i,pq} X_{pq} = Tr(T_i^T X).
    # Portanto, para medir Tr(M_i X) com M_i Hermitiano, utiliza-se T_i = M_i^T via swapaxes.
    A = DenseConstraintOp(np.swapaxes(M_full, -1, -2), dom, cod, ctx)

    # Custo zero (C = 0) para o problema de factibilidade estrita
    sdp = SDPProblem(dom.zeros(), A, b_full, ctx=ctx)

    return sdp, M_full, b_full


# ===========================================================================
# 4. Resolução do problema com SDPLab (run_cvxpy_solver)
# ===========================================================================

def solve_qmp(
    sdp: SDPProblem,
    solver: str = "CLARABEL",
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""
    Resolve o problema SDP de representabilidade de marginais utilizando
    o backend de resolução de SDPLab (`sdplab.solvers.run_cvxpy_solver`).

    Parâmetros:
    -----------
    sdp : SDPProblem
        Problema montado pela função `build_qmp_sdp`.
    solver : str, padrão "CLARABEL"
        Nome do solver cônico (ex: "CLARABEL", "SCS").
    verbose : bool, padrão False
        Imprime logs do solver durante a resolução.

    Retorno:
    --------
    Dict contendo:
        - "status": status retornado pelo solver ("optimal", "infeasible", etc.)
        - "is_feasible": booleano indicando se o problema é factível
        - "rho": matriz densidade global recuperada (np.ndarray ou None)
        - "dual_y": multiplicadores de Lagrange / variáveis duais
        - "problem": objeto cvxpy.Problem correspondente
        - "error_message": mensagem de erro caso o solver falhe ou detecte infactibilidade
    """
    try:
        X, y, prob = run_cvxpy_solver(
            sdp,
            solver=solver,
            verbose=verbose,
            return_problem=True,
            **kwargs,
        )
        status = prob.status
        is_feasible = (status in ("optimal", "optimal_inaccurate"))
        rho_mat = np.asarray(X)
        # Garante simetria hermitiana perfeita numérica
        rho_mat = (rho_mat + rho_mat.conj().T) / 2.0

        return {
            "status": status,
            "is_feasible": is_feasible,
            "rho": rho_mat,
            "dual_y": np.asarray(y),
            "problem": prob,
            "error_message": None,
        }
    except ValueError as err:
        err_str = str(err)
        status = "infeasible" if "infeasible" in err_str.lower() else "failed"
        return {
            "status": status,
            "is_feasible": False,
            "rho": None,
            "dual_y": None,
            "problem": None,
            "error_message": err_str,
        }


# ===========================================================================
# 5. Verificação da solução e cálculo de resíduos e erros
# ===========================================================================

def verify_marginals(
    rho: Optional[np.ndarray],
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    r"""
    Realiza a auditoria numérica completa da matriz densidade candidata :math:`\rho`:
    1. Erro de Hermiticidade: :math:`\|\rho - \rho^\dagger\|_F`.
    2. Erro de Normalização: :math:`|\operatorname{Tr}(\rho) - 1|`.
    3. Condição de Positividade: :math:`\rho \succeq 0` via :math:`\lambda_{\min}(\rho) \ge -\mathrm{tol}`.
    4. Espectro completo: todos os autovalores de :math:`\rho`.
    5. Pureza: :math:`\gamma = \operatorname{Tr}(\rho^2)`.
    6. Comparação com as marginais fornecidas: erro Frobenius e erro máximo elemento a elemento.
    """
    if rho is None:
        return {"is_feasible": False, "message": "Nenhuma matriz rho fornecida (problema infactível)."}

    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        num_qubits = int(round(np.log2(rho.shape[0])))
        dims = [2] * num_qubits
    dims = tuple(dims)

    # 1. Hermiticidade
    herm_err = float(np.linalg.norm(rho - rho.conj().T, "fro"))

    # 2. Normalização
    tr_val = complex(np.trace(rho))
    tr_err = float(abs(tr_val - 1.0))

    # 3. Autovalores e positividade
    rho_sym = (rho + rho.conj().T) / 2.0
    evals = np.linalg.eigvalsh(rho_sym)
    min_eval = float(np.min(evals))
    is_psd = bool(min_eval >= -tol)

    # 4. Pureza
    purity = float(np.real(np.trace(rho_sym @ rho_sym)))

    # 5. Verificação das marginais
    marginal_errors: Dict[str, Dict[str, Any]] = {}
    for k, omega in marginals_dict.items():
        sub_tuple = parse_subsystem_key(k)
        omega_np = np.asarray(omega, dtype=complex)
        calc_marginal = partial_trace(rho_sym, keep=sub_tuple, dims=dims)
        diff = calc_marginal - omega_np
        fro_err = float(np.linalg.norm(diff, "fro"))
        max_err = float(np.max(np.abs(diff)))
        marginal_errors[str(k)] = {
            "frobenius_error": fro_err,
            "max_abs_error": max_err,
            "calculated_marginal": calc_marginal,
            "target_marginal": omega_np,
        }

    return {
        "is_feasible": True,
        "hermitian_error": herm_err,
        "trace": tr_val,
        "trace_error": tr_err,
        "eigenvalues": evals,
        "min_eigenvalue": min_eval,
        "is_psd": is_psd,
        "purity": purity,
        "marginal_errors": marginal_errors,
    }


# ===========================================================================
# 6. Execução dos três exemplos canônicos do projeto
# ===========================================================================

def run_all_examples(solver: str = "CLARABEL") -> None:
    """
    Executa e exibe os resultados dos três exemplos definidos no projeto.
    """
    sep = "=" * 78
    sub_sep = "-" * 78

    print(sep)
    print("QUANTUM MARGINAL PROBLEM (QMP) VIA SDPLab")
    print(f"Versão SDPLab: {sdplab.__version__} | Versão SpaceCore: {spacecore.__version__}")
    print(f"Solver cônico utilizado via SDPLab: {solver}")
    print(sep)

    # -----------------------------------------------------------------------
    # Exemplo 1: Marginais do estado GHZ
    # -----------------------------------------------------------------------
    print("\n" + sep)
    print("EXEMPLO 1: Marginais do estado |GHZ_theta>")
    print("omega_AB = omega_AC = omega_BC = 1/2 (|00><00| + |11><11|)")
    print("Regime esperado: mixed = SIM, pure = SIM")
    print(sub_sep)

    omega_ghz = 0.5 * (ketbra(np.kron(zero, zero)) + ketbra(np.kron(one, one)))
    marginals_ghz = {
        "AB": omega_ghz,
        "AC": omega_ghz,
        "BC": omega_ghz,
    }

    sdp_ghz, M_ghz, b_ghz = build_qmp_sdp(marginals_ghz, dims=[2, 2, 2])
    print(f"SDP montado com sucesso: dim(dom) = {sdp_ghz.dom.n}x{sdp_ghz.dom.n}, restrições m = {len(b_ghz)}")

    res_ghz = solve_qmp(sdp_ghz, solver=solver)
    print(f"Status do solver: {res_ghz['status']} (Factível: {res_ghz['is_feasible']})")

    verif_ghz = verify_marginals(res_ghz["rho"], marginals_ghz, dims=[2, 2, 2])
    print(f"Erro de Hermiticidade ||rho - rho^dagger||_F: {verif_ghz['hermitian_error']:.2e}")
    print(f"Traço Tr(rho): {verif_ghz['trace'].real:.12f} (Erro: {verif_ghz['trace_error']:.2e})")
    print(f"Menor autovalor lambda_min: {verif_ghz['min_eigenvalue']:.2e} -> rho >= 0: {verif_ghz['is_psd']}")
    print(f"Pureza Tr(rho^2): {verif_ghz['purity']:.4f}")
    print("Autovalores de rho_ABC:")
    print(" ", np.round(verif_ghz["eigenvalues"], 6))
    print("Resíduos das marginais:")
    for k, err in verif_ghz["marginal_errors"].items():
        print(f"  Marginal {k:2s}: Erro Frobenius = {err['frobenius_error']:.2e}, Erro Max Abs = {err['max_abs_error']:.2e}")

    # -----------------------------------------------------------------------
    # Exemplo 2: Marginais Maximamente Mistas
    # -----------------------------------------------------------------------
    print("\n" + sep)
    print("EXEMPLO 2: Marginais Maximamente Mistas")
    print("omega_AB = omega_AC = omega_BC = I_4 / 4")
    print("Regime esperado: mixed = SIM, pure = NÃO")
    print(sub_sep)

    omega_mixed = np.eye(4, dtype=complex) / 4.0
    marginals_mixed = {
        "AB": omega_mixed,
        "AC": omega_mixed,
        "BC": omega_mixed,
    }

    sdp_mixed, M_mixed, b_mixed = build_qmp_sdp(marginals_mixed, dims=[2, 2, 2])
    print(f"SDP montado com sucesso: dim(dom) = {sdp_mixed.dom.n}x{sdp_mixed.dom.n}, restrições m = {len(b_mixed)}")

    res_mixed = solve_qmp(sdp_mixed, solver=solver)
    print(f"Status do solver: {res_mixed['status']} (Factível: {res_mixed['is_feasible']})")

    verif_mixed = verify_marginals(res_mixed["rho"], marginals_mixed, dims=[2, 2, 2])
    print(f"Erro de Hermiticidade ||rho - rho^dagger||_F: {verif_mixed['hermitian_error']:.2e}")
    print(f"Traço Tr(rho): {verif_mixed['trace'].real:.12f} (Erro: {verif_mixed['trace_error']:.2e})")
    print(f"Menor autovalor lambda_min: {verif_mixed['min_eigenvalue']:.2e} -> rho >= 0: {verif_mixed['is_psd']}")
    print(f"Pureza Tr(rho^2): {verif_mixed['purity']:.4f} (Esperado para I_8/8: 1/8 = 0.1250)")
    print("Autovalores de rho_ABC:")
    print(" ", np.round(verif_mixed["eigenvalues"], 6))
    print("Resíduos das marginais:")
    for k, err in verif_mixed["marginal_errors"].items():
        print(f"  Marginal {k:2s}: Erro Frobenius = {err['frobenius_error']:.2e}, Erro Max Abs = {err['max_abs_error']:.2e}")

    # -----------------------------------------------------------------------
    # Exemplo 3: Incompatibilidade (Bell em AB e AC)
    # -----------------------------------------------------------------------
    print("\n" + sep)
    print("EXEMPLO 3: Incompatibilidade de Marginais")
    print("omega_AB = |Phi+><Phi+|, omega_AC = |Phi+><Phi+|")
    print("Regime esperado: mixed = NÃO, pure = NÃO (Monogamia do entrelaçamento)")
    print(sub_sep)

    phi_plus = bell_phi_plus()
    rho_bell = ketbra(phi_plus)
    marginals_incomp = {
        "AB": rho_bell,
        "AC": rho_bell,
    }

    sdp_incomp, M_incomp, b_incomp = build_qmp_sdp(marginals_incomp, dims=[2, 2, 2])
    print(f"SDP montado com sucesso: dim(dom) = {sdp_incomp.dom.n}x{sdp_incomp.dom.n}, restrições m = {len(b_incomp)}")

    res_incomp = solve_qmp(sdp_incomp, solver=solver)
    print(f"Status retornado pelo solver: {res_incomp['status']}")
    print(f"O problema é factível? {res_incomp['is_feasible']}")
    print(f"Diagnóstico do backend SDPLab:")
    print(f"  Mensagem de exceção capturada: {res_incomp['error_message']}")
    print(
        "  Explicação matemática da falha: O solver cônico detectou que o sistema primal é infactível,\n"
        "  produzindo um certificado dual de infactibilidade (raio de Farkas). Fisicamente, pela monogamia\n"
        "  do entrelaçamento e pela forte subaditividade da entropia de von Neumann, um qubit não pode estar\n"
        "  simultaneamente em um estado maximamente entrelaçado puro com dois subsistemas distintos."
    )
    print(sep)


if __name__ == "__main__":
    solver_choice = "CLARABEL"
    if len(sys.argv) > 1:
        solver_choice = sys.argv[1].upper()
    run_all_examples(solver=solver_choice)
