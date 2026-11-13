from .operadores import base_hermitiana, embed_operator
from .validacoes import validar_dimensoes, validar_marginais
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

def restricoes_lineares(
        marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
        dims: Sequence[int]
):
    dims_tuple = validar_dimensoes(dims)
    normalized_marginals = validar_marginais(marginals_dict, dims_tuple)
    d_total = int(np.prod(dims_tuple))

    M_list: List[np.ndarray] = []
    b_list: List[float] = []

    # 2. Restrição de normalização: Tr(rho) = Tr(I_d rho) = 1.0
    
    M_list.append(np.eye(d_total, dtype=complex))
    b_list.append(1.0)

    # 3. Restrições afins das marginais
    for sistema, omega_S in normalized_marginals.items():
        d_S = omega_S.shape[0]
        base_S = base_hermitiana(d_S)
        for E in base_S:
            val = float(np.real(np.trace(E @ omega_S)))
            M_global = embed_operator(E, sistema, dims_tuple)
            M_list.append(M_global)
            b_list.append(val)

    M_full = np.stack(M_list, axis=0)  # Formato (m, d_total, d_total)
    b_full = np.array(b_list, dtype=float)

    return M_full, b_full
