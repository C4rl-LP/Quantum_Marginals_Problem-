import numpy as np
from .validacoes import (
    validar_escalar_positivo,
    validar_parametros_otimizacao,
)


def otimizar_dual(
    problema: "DualEntropico",
    u_inicial=None,
    tolerancia=1e-6,
    maxiter=2000,
    metodo="gradient",
    salvar_matriz_densidade=True,
    **kwargs
):
    """Maximiza o funcional dual entrópico."""

    maximizadores = {
        "gradient": gradient_acendent,
        "stochastic": gradient_acendent_sthocastical
    }

    if metodo not in maximizadores:
        raise ValueError(
            f"Método '{metodo}' desconhecido. "
            f"Métodos disponíveis: {list(maximizadores)}"
        )

    maximizador = maximizadores[metodo]

    if u_inicial is None:
        u_inicial = np.zeros(len(problema.Q))
    u_inicial, tolerancia, maxiter = validar_parametros_otimizacao(
        u_inicial, len(problema.Q), tolerancia, maxiter
    )


    u_otimo, historico = maximizador(
        problema,
        u_0=u_inicial,
        maxinter=maxiter,
        tol=tolerancia,
        **kwargs
    )
    valor_dual, gradiente = problema.avaliar(u_otimo)
    residuos = -gradiente
    norma_residuos = float(np.linalg.norm(residuos))
    residuo_maximo = float(np.linalg.norm(residuos, ord=np.inf))
    resultado = {
        "metodo": metodo,
        "multiplicadores": u_otimo,
        "valor_dual": valor_dual,
        "gradiente": gradiente,
        "residuos": residuos,
        "norma_residuos": norma_residuos,
        "residuo_maximo": residuo_maximo,
        "convergiu": norma_residuos < tolerancia,
        "historico": historico
    }
    if salvar_matriz_densidade:
        resultado["estado"] = problema.estado(u_otimo)
    return resultado



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


def varrer_dual(criar_problema, epsilons, *, fator_learnrate,
                reutilizar_multiplicadores=False, u_inicial=None, **opcoes):
    """Gera um resultado por ε na ordem recebida, inclusive sem convergência.

    u_inicial pode ser um vetor ou uma função que recebe o problema. Com
    reutilização, a inicialização só é usada no primeiro ponto.
    """
    for epsilon in epsilons:
        problema = criar_problema(epsilon)
        inicio = u_inicial(problema) if callable(u_inicial) else u_inicial
        resultado = otimizar_dual(
            problema, u_inicial=inicio, learnrate=fator_learnrate * epsilon, **opcoes,
        )
        if reutilizar_multiplicadores:
            u_inicial = resultado["multiplicadores"].copy()
        yield resultado


def varrer_taxas_dual(criar_problema, epsilons, fatores, **opcoes):
    """Retorna a matriz de resíduos finais com uma linha por fator e uma coluna por ε."""
    epsilons, fatores = tuple(epsilons), tuple(fatores)
    residuos = np.empty((len(fatores), len(epsilons)))
    for j, epsilon in enumerate(epsilons):
        problema = criar_problema(epsilon)
        for i, fator in enumerate(fatores):
            resultado = otimizar_dual(
                problema, learnrate=fator * epsilon,
                salvar_matriz_densidade=False, **opcoes,
            )
            residuos[i, j] = resultado["norma_residuos"]
    return residuos
