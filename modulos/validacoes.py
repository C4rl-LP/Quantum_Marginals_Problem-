"""Validação de entradas do dual e das marginais.

As funções retornam entradas convertidas e lançam ValueError em caso de erro.
Não certificam compatibilidade global nem convergência do solver.
"""

from collections.abc import Mapping
from numbers import Integral
import numpy as np


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


def validar_hermitiana(valor, nome="matriz", dimensao=None, tol=1e-10):
    """Exige matriz quadrada não vazia, finita e hermitiana (erro absoluto)."""
    tol = validar_escalar_positivo(tol, "tol")
    a = np.array(valor, dtype=complex, copy=True)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{nome} deve ser uma matriz quadrada não vazia.")
    if dimensao is not None and a.shape != (dimensao, dimensao):
        raise ValueError(f"{nome} deve ter formato {(dimensao, dimensao)}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{nome} deve conter apenas valores finitos.")
    if not np.allclose(a, a.conj().T, atol=tol, rtol=0):
        raise ValueError(f"{nome} deve ser hermitiana.")
    return a


def validar_densidade(valor, nome="estado", dimensao=None, tol=1e-10):
    """Exige matriz densidade PSD e de traço um; aceita estados singulares."""
    a = validar_hermitiana(valor, nome, dimensao, tol)
    if abs(np.trace(a) - 1) > tol:
        raise ValueError(f"{nome} deve ter traço um.")
    if np.linalg.eigvalsh(a / 2 + a.conj().T / 2).min() < -tol:
        raise ValueError(f"{nome} deve ser positiva semidefinida.")
    return a


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
