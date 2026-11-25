"""Validação de entradas e diagnósticos numéricos de estados e marginais."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .operadores import reduzir_marginais


def validar_escalar_positivo(valor, nome="valor"):
    """Exige um escalar real, finito e estritamente positivo."""
    a = np.asarray(valor)
    if a.ndim != 0 or a.dtype.kind not in "iuf" or not np.isfinite(a) or a <= 0:
        raise ValueError(f"{nome} deve ser um escalar real finito e positivo.")
    return float(a)


def validar_vetor_real(valor, nome="vetor", tamanho=None):
    """Converte um vetor real finito e verifica seu comprimento opcional."""
    a = np.asarray(valor)
    if a.ndim != 1 or a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)):
        raise ValueError(f"{nome} deve ser um vetor real finito.")
    if tamanho is not None and len(a) != tamanho:
        raise ValueError(f"{nome} deve ter comprimento {tamanho}; recebido {len(a)}.")
    return np.array(a, dtype=float, copy=True)


def _matriz_quadrada_finita(valor, nome):
    matriz = np.asarray(valor, dtype=complex)
    if matriz.ndim != 2 or matriz.shape[0] == 0 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError(f"{nome} deve ser uma matriz quadrada não vazia.")
    if not np.isfinite(matriz).all():
        raise ValueError(f"{nome} deve conter apenas valores finitos.")
    return matriz


def _parte_hermitiana(matriz):
    return matriz / 2 + matriz.conj().T / 2


def _eh_hermitiana(matriz, tol):
    return bool(np.allclose(matriz, matriz.conj().T, atol=tol, rtol=0))


def validar_hermitiana(valor, nome="matriz", dimensao=None, tol=1e-10):
    """Exige matriz quadrada não vazia, finita e hermitiana (erro absoluto)."""
    tol = validar_escalar_positivo(tol, "tol")
    matriz = _matriz_quadrada_finita(valor, nome)
    if dimensao is not None and matriz.shape != (dimensao, dimensao):
        raise ValueError(f"{nome} deve ter formato {(dimensao, dimensao)}.")
    if not _eh_hermitiana(matriz, tol):
        raise ValueError(f"{nome} deve ser hermitiana.")
    return matriz.copy()


def validar_densidade(valor, nome="estado", dimensao=None, tol=1e-10):
    """Exige matriz densidade PSD e de traço um; aceita estados singulares."""
    matriz = validar_hermitiana(valor, nome, dimensao, tol)
    if abs(np.trace(matriz) - 1) > tol:
        raise ValueError(f"{nome} deve ter traço um.")
    if np.linalg.eigvalsh(_parte_hermitiana(matriz))[0] < -tol:
        raise ValueError(f"{nome} deve ser positiva semidefinida.")
    return matriz


def validar_dimensoes(dims):
    """Exige sequência não vazia de dimensões inteiras positivas."""
    dims = tuple(dims)
    if not dims or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, Integral)
                       or d <= 0 for d in dims):
        raise ValueError("dims deve conter inteiros positivos e não pode ser vazia.")
    return tuple(int(d) for d in dims)


def validar_subsistema(chave, dims):
    """Valida letras A–Z ou índices, preservando a ordem tensorial informada."""
    dims = validar_dimensoes(dims)
    if isinstance(chave, str):
        texto = chave.strip().upper()
        if not texto or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in texto):
            raise ValueError(f"Chave de subsistema inválida: {chave!r}.")
        indices = tuple(ord(c) - ord('A') for c in texto)
    elif isinstance(chave, (tuple, list)):
        indices = tuple(chave)
    else:
        raise ValueError("O subsistema deve ser uma string ou sequência de índices.")
    if (not indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                           or i < 0 or i >= len(dims) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError(f"Índices inválidos ou repetidos no subsistema {chave!r}.")
    return tuple(int(i) for i in indices)


def validar_marginais(marginais, dims, tol=1e-10):
    """Normaliza chaves e valida estados locais, rejeitando chaves equivalentes.

    A ordem das letras/índices deve corresponder à ordem dos fatores na matriz.
    Esta checagem local não decide se existe uma extensão global.
    """
    dims = validar_dimensoes(dims)
    if not isinstance(marginais, Mapping):
        raise ValueError("marginais deve ser um mapeamento de subsistemas para matrizes.")
    resultado = {}
    for chave, matriz in marginais.items():
        indices = validar_subsistema(chave, dims)
        if indices in resultado:
            raise ValueError(f"Subsistema duplicado: {chave!r} equivale a {indices}.")
        dimensao = int(np.prod([dims[i] for i in indices]))
        resultado[indices] = validar_densidade(matriz, f"Marginal {chave!r}", dimensao, tol)
    return resultado


def validar_dados_dual(H, Q, q, epsilon):
    """Valida operadores globais, alvos e regularização; retorna cópias numéricas."""
    H = validar_hermitiana(H, "H")
    Q = tuple(validar_hermitiana(Qi, f"Q[{i}]", H.shape[0])
              for i, Qi in enumerate(Q))
    q = validar_vetor_real(q, "q", len(Q))
    epsilon = validar_escalar_positivo(epsilon, "epsilon")
    return H, Q, q, epsilon


def validar_parametros_otimizacao(u_inicial, quantidade, tolerancia, maxiter):
    """Valida o vetor inicial, tolerância positiva e limite inteiro de iterações."""
    u = validar_vetor_real(u_inicial, "u_inicial", quantidade)
    tolerancia = validar_escalar_positivo(tolerancia, "tolerancia")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, Integral) or maxiter <= 0:
        raise ValueError("maxiter deve ser um inteiro positivo.")
    return u, tolerancia, int(maxiter)


def _diagnosticar_densidade(rho, tol):
    tol = validar_escalar_positivo(tol, "tol")
    rho = _matriz_quadrada_finita(rho, "estado")
    simetrizada = _parte_hermitiana(rho)
    autovalores = np.linalg.eigvalsh(simetrizada)
    traco = complex(np.trace(rho))
    return simetrizada, {
        "hermitiana": _eh_hermitiana(rho, tol),
        "hermitian_error": float(np.linalg.norm(rho - rho.conj().T, "fro")),
        "trace": traco,
        "trace_error": float(abs(traco - 1)),
        "eigenvalues": autovalores,
        "min_eigenvalue": float(autovalores[0]),
        "is_psd": bool(autovalores[0] >= -tol),
    }


def verifica_densidade(rho: np.ndarray, tol: float = 1e-10) -> Dict[str, Any]:
    """Diagnostica hermiticidade, positividade e traço com tolerância absoluta."""
    _, dados = _diagnosticar_densidade(rho, tol)
    return {
        "hermitiana": dados["hermitiana"],
        "positiva": dados["is_psd"],
        "traco_1": dados["trace_error"] <= tol,
        "autovalores": dados["eigenvalues"],
    }


def _verify_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> Dict[str, Any]:
    """Mantém o formato histórico de diagnóstico booleano e erros."""
    _, dados = _diagnosticar_densidade(rho, tol)
    return {
        "hermitian_error": dados["hermitian_error"],
        "trace_error": dados["trace_error"],
        "min_eigenvalue": dados["min_eigenvalue"],
        "is_valid": dados["hermitiana"] and dados["is_psd"] and dados["trace_error"] <= tol,
    }


def comparar_marginais(estado, marginais, dims):
    """Retorna a redução, o alvo e os erros de Frobenius e máximo por chave."""
    diagnosticos = {}
    for chave, reducao in reduzir_marginais(estado, marginais, dims).items():
        alvo = np.asarray(marginais[chave], dtype=complex)
        diferenca = reducao - alvo
        diagnosticos[chave] = {
            "frobenius_error": float(np.linalg.norm(diferenca, "fro")),
            "max_abs_error": float(np.max(np.abs(diferenca))),
            "calculated_marginal": reducao,
            "target_marginal": alvo,
        }
    return diagnosticos


def verify_marginals(
    rho: Optional[np.ndarray],
    marginals_dict: Dict[Union[str, Tuple[int, ...]], np.ndarray],
    dims: Optional[Sequence[int]] = None,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    """Compara reduções de ``rho`` com as marginais-alvo e retorna diagnósticos.

    O campo ``is_feasible`` informa se uma matriz foi fornecida; os erros e
    propriedades físicas são reportados separadamente no restante do resultado.
    """
    if rho is None:
        return {"is_feasible": False, "message": "Nenhuma matriz rho fornecida (problema infactível)."}

    rho_sym, dados = _diagnosticar_densidade(rho, tol)
    if dims is None:
        dims = (2,) * int(round(np.log2(rho_sym.shape[0])))
    dados.pop("hermitiana")

    marginal_errors = {
        str(chave): comparacao
        for chave, comparacao in comparar_marginais(rho_sym, marginals_dict, tuple(dims)).items()
    }

    return {
        "is_feasible": True,
        **dados,
        "purity": float(np.trace(rho_sym @ rho_sym).real),
        "marginal_errors": marginal_errors,
    }
