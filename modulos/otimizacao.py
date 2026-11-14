import numpy as np
from scipy.optimize import minimize
from .validacoes import validar_parametros_otimizacao


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



def gradient_descent(problema,u_0 : np.ndarray,learnrate:float,  maxinter =2000, tol = 1e-8,  ):
    tau = learnrate
    _, grad = problema.avaliar(u_0)
    u_now = u_0
    resudou = np.linalg.norm(grad)
    for i in range(len(maxinter)):
        
        u_new = u_now - tau*grad
        u_now = u_new
        grad = problema.avaliar(u_new)
        resudou = np.linalg.norm(grad)
        if resudou < tol:
            break

    return u_new, 
