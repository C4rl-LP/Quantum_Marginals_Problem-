import numpy as np
from scipy.optimize import minimize
from .validacoes import (
    validar_escalar_positivo,
    validar_parametros_otimizacao,
    validar_vetor_real,
)


def otimizar_dual(
    problema,
    u_inicial=None,
    tolerancia=1e-8,
    maxiter=2000
):
    """Maximiza o funcional dual entrópico."""

    if u_inicial is None:
        u_inicial = np.zeros(len(problema.Q))
    u_inicial, tolerancia, maxiter = validar_parametros_otimizacao(
        u_inicial, len(problema.Q), tolerancia, maxiter
    )

    def objetivo(u):
        valor, gradiente = problema.avaliar(u)
        return -valor, -gradiente

    resultado = minimize(
        objetivo,
        u_inicial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": maxiter,
            "gtol": tolerancia
        }
    )

    u_otimo = resultado.x
    gamma = problema.estado(u_otimo)
    valor_dual, gradiente = problema.avaliar(u_otimo)
    residuos = -gradiente

    return {
        "multiplicadores": u_otimo,
        "estado": gamma,
        "valor_dual": valor_dual,
        "residuos": residuos,
        "convergiu": resultado.success,
        "resultado_scipy": resultado
    }



def gradient_acendent(
    problema, u_0: np.ndarray = None, learnrate: float = 1e-2,
    maxinter: int = 2000, tol: float = 1e-8,
):
    """Maximiza o dual por subida de gradiente em todas as coordenadas.

    O histórico contém pares (valor_dual, norma_do_gradiente), avaliados
    antes de cada atualização.
    """
    quantidade = len(problema.q)
    if u_0 is None:
        u_0 = np.zeros(quantidade, dtype=float)
    u, tol, maxinter = validar_parametros_otimizacao(
        u_0, quantidade, tol, maxinter
    )
    tau = validar_escalar_positivo(learnrate, "learnrate")
    historico = []

    for _ in range(maxinter):
        valor, grad = problema.avaliar(u)
        residuo = float(np.linalg.norm(grad))
        historico.append((valor, residuo))
        if residuo < tol:
            break
        u += tau * grad

    return u, historico


def gradient_acendent_sthocastical(
    problema, u_0: np.ndarray = None, dim_choice: int = 1,
    learnrate: float = 1e-2, step_check: int = 100,
    maxinter: int = 2000, tol: float = 1e-8,
):
   
    quantidade = len(problema.q)
    if isinstance(dim_choice, (bool, np.bool_)) or not isinstance(dim_choice, (int, np.integer)) or not 1 <= dim_choice <= quantidade:
        raise ValueError("dim_choice deve ser um inteiro entre 1 e len(problema.q).")
    if isinstance(step_check, (bool, np.bool_)) or not isinstance(step_check, (int, np.integer)) or step_check <= 0:
        raise ValueError("step_check deve ser um inteiro positivo.")
    if u_0 is None:
        u_0 = np.zeros(quantidade, dtype=float)
    u, tol, maxinter = validar_parametros_otimizacao(
        u_0, quantidade, tol, maxinter
    )

    tau = validar_escalar_positivo(learnrate, "learnrate")
    
    indices = np.arange(quantidade)
    historico = []

    for i in range(maxinter):
        
        if i % step_check == 0:
            valor, grad_completo = problema.avaliar(u)
            residuo_completo = float(np.linalg.norm(grad_completo))
            if residuo_completo < tol:
                historico.append((valor, residuo_completo))
                break

        mask = np.random.choice(indices, size=dim_choice, replace=False)
        valor, grad = problema.avaliar(u, mask)
        residuo = float(np.linalg.norm(grad))
        historico.append((valor, residuo))
        u[mask] += tau * grad

    return u, historico
