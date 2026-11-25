"""Estimativas espectrais e certificados para problemas de reconstrução."""

import numpy as np

from .estados import entropia_von_neumann
from .validacoes import validar_escalar_positivo, validar_hermitiana


def gap_espectral(hamiltoniano):
    """Retorna a diferença entre os dois menores autovalores, com multiplicidade."""
    energias = np.linalg.eigvalsh(validar_hermitiana(hamiltoniano, "Hamiltoniano"))
    if energias.size < 2:
        raise ValueError("O Hamiltoniano deve ter pelo menos dois níveis.")
    return float(energias[1] - energias[0])


def cota_frobenius(epsilons, gap, marginais):
    """Calcula a cota para a solução ótima factível com referência fundamental pura."""
    gap = validar_escalar_positivo(gap, "gap")
    epsilons = np.asarray(epsilons, dtype=float)
    if not np.isfinite(epsilons).all() or np.any(epsilons <= 0):
        raise ValueError("Os valores de epsilon devem ser positivos e finitos.")
    soma_entropias = sum(entropia_von_neumann(rho) for rho in marginais.values())
    return np.sqrt(2 * epsilons * soma_entropias / gap)


def avaliar_testemunha(operador, alvo):
    """Retorna o máximo físico e a separação entre o alvo e esse máximo.

    Uma separação positiva certifica que o valor esperado prescrito é inviável.
    O operador deve ser hermitiano e estar no espaço global.
    """
    maximo = float(np.linalg.eigvalsh(validar_hermitiana(operador, "testemunha"))[-1])
    return maximo, alvo - maximo
