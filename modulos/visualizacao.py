"""Visualizações de estados quânticos reduzidos."""

from collections.abc import Sequence
from operator import index
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

from .operadores import parse_subsystem_key, partial_trace


def _wigner_base_fock(dim: int, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Avalia a Wigner de cada operador |m><n| na grade (p, x)."""
    z = np.sqrt(2.0) * (x[None, :] + 1j * p[:, None])
    raio2 = np.abs(z) ** 2
    gaussiana = np.exp(-raio2 / 2) / np.pi
    log_fatoriais = gammaln(np.arange(1, dim + 1))
    base = np.empty((dim, dim, p.size, x.size), dtype=complex)
    for m in range(dim):
        for n in range(m, dim):
            ordem = n - m
            fator = np.exp((log_fatoriais[m] - log_fatoriais[n]) / 2)
            base[m, n] = (
                (-1) ** m * gaussiana * fator * z ** ordem
                * eval_genlaguerre(m, ordem, raio2)
            )
            if n != m:
                base[n, m] = base[m, n].conj()
    return base


def wigner_marginal(
    rho: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    keep: Sequence[int] | None = None,
    trace_out: Sequence[int] | None = None,
    dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Calcula a Wigner conjunta dos modos selecionados na base de Fock.

    Informe exatamente um entre ``keep`` e ``trace_out``. Sem ``dims``,
    assume dimensões locais 2, como ``partial_trace``. Os modos retidos
    seguem a ordem de ``dims`` e usam as mesmas grades 1D ``x`` e ``p``.
    Os eixos do resultado são ``(p_i, x_i, p_j, x_j, ...)`` na convenção
    hbar = 1 do QuTiP. Sem modos retidos, retorna o traço como array 0D.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] == 0:
        raise ValueError("rho deve ser uma matriz quadrada não vazia.")
    if dims is None:
        dims = (2,) * (rho.shape[0].bit_length() - 1)
    dims = tuple(map(index, dims))
    if any(dim < 1 for dim in dims) or np.prod(dims, dtype=object) != rho.shape[0]:
        raise ValueError("As dimensões locais devem ser positivas e compatíveis com rho.")
    if (keep is None) == (trace_out is None):
        raise ValueError("Especifique exatamente um entre keep e trace_out.")

    selecionados = tuple(map(index, keep if keep is not None else trace_out))
    modos = set(range(len(dims)))
    if len(set(selecionados)) != len(selecionados) or not modos.issuperset(selecionados):
        raise ValueError("Os índices dos subsistemas devem ser válidos e distintos.")
    retidos = sorted(selecionados if keep is not None else modos.difference(selecionados))
    reduzida = partial_trace(rho, keep=retidos, dims=dims)
    if not retidos:
        return reduzida.real.reshape(())

    x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or not x.size or not p.size:
        raise ValueError("x e p devem ser grades unidimensionais não vazias.")
    dims_reduzidas = tuple(dims[i] for i in retidos)
    if len(retidos) == 1:
        from qutip import Qobj, wigner

        return wigner(Qobj(reduzida), x, p)

    # Cada contração substitui os dois índices de Fock do modo por (p, x).
    wigner_conjunta = reduzida.reshape(dims_reduzidas + dims_reduzidas)
    bases = {dim: _wigner_base_fock(dim, x, p) for dim in set(dims_reduzidas)}
    for dim, restantes in zip(dims_reduzidas, range(len(retidos), 0, -1)):
        wigner_conjunta = np.tensordot(wigner_conjunta, bases[dim], axes=((0, restantes), (0, 1)))
    return wigner_conjunta.real


def cortes_wigner(
    rho: np.ndarray,
    grade: np.ndarray,
    keep: Sequence[int],
    dims: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna os cortes de posição e de momento da Wigner conjunta de dois modos.

    O corte de posição fixa os momentos em zero e o de momento fixa as
    posições em zero. Em cada matriz, as colunas correspondem ao primeiro
    modo retido na ordem de dims e as linhas ao segundo.
    """
    if len(keep) != 2:
        raise ValueError("Informe dois subsistemas distintos e válidos.")

    posicoes = wigner_marginal(rho, grade, [0.0], keep=keep, dims=dims)
    momentos = wigner_marginal(rho, [0.0], grade, keep=keep, dims=dims)
    return posicoes[0, :, 0, :].T, momentos[:, 0, :, 0].T


def _base_posicao(dim: int, x: np.ndarray) -> np.ndarray:
    """Avalia as funções de onda normalizadas dos níveis de Fock."""
    base = np.empty((dim, x.size))
    base[0] = np.exp(-x**2 / 2) / np.pi**0.25
    if dim > 1:
        base[1] = np.sqrt(2) * x * base[0]
    for n in range(2, dim):
        base[n] = np.sqrt(2 / n) * x * base[n - 1] - np.sqrt((n - 1) / n) * base[n - 2]
    return base


def distribuicao_conjunta_posicoes(rho, x, keep, dims):
    """Avalia a marginal de posições da Wigner de dois modos na grade x.

    Retorna P(x_i, x_j), com os eixos na ordem dos subsistemas em dims.
    A projeção na base de posição equivale à integração sobre os momentos.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not x.size:
        raise ValueError("x deve ser uma grade unidimensional não vazia.")
    i, j = sorted(map(index, keep))
    if not 0 <= i < j < len(dims):
        raise ValueError("Informe dois subsistemas distintos e válidos.")
    reduzida = partial_trace(rho, keep=(i, j), dims=dims)
    tensor = reduzida.reshape(dims[i], dims[j], dims[i], dims[j])
    base_i = _base_posicao(dims[i], x)
    base_j = _base_posicao(dims[j], x)
    return np.einsum("ax,by,abcd,cx,dy->xy", base_i, base_j, tensor, base_i, base_j,
                     optimize=True).real


def visualizar_marginais(
    marginals_dict: dict[str | tuple[int, ...], np.ndarray],
    limite: float = 5.0,
    pontos: int = 101,
) -> tuple[plt.Figure, np.ndarray]:
    """Plota a função de Wigner das marginais de um único modo."""
    if not marginals_dict:
        raise ValueError("O dicionário de marginais não pode estar vazio.")

    chaves = list(marginals_dict)
    if any(len(parse_subsystem_key(chave)) != 1 for chave in chaves):
        raise ValueError("A visualização de Wigner espera marginais de um subsistema por chave.")

    x = np.linspace(-limite, limite, pontos)
    fig, axes = plt.subplots(1, len(chaves), figsize=(5 * len(chaves), 4), squeeze=False)

    for ax, chave in zip(axes[0], chaves):
        matriz = np.asarray(marginals_dict[chave], dtype=complex)
        W = wigner_marginal(matriz, x, x, keep=(0,), dims=(matriz.shape[0],))
        imagem = ax.contourf(x, x, W, levels=40, cmap="RdBu_r")
        ax.set_title(f"Marginal {chave}")
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        fig.colorbar(imagem, ax=ax, label="W(x, p)")

    fig.tight_layout()
    return fig, axes[0]


def salvar_figura(figura, caminho, *, dpi=200):
    """Cria a pasta de destino, grava e fecha a figura e retorna o caminho."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=dpi, facecolor="white")
    plt.close(figura)
    return caminho


def visualizar_matriz_densidade(eixo, matriz, titulo, *, base=None, vmax=0.5):
    """Desenha a parte real da matriz quadrada com anotações na base fornecida."""
    matriz = np.asarray(matriz)
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("A matriz deve ser quadrada.")
    indices = range(len(matriz))
    base = tuple(indices) if base is None else tuple(base)
    imagem = eixo.imshow(matriz.real, vmin=0, vmax=vmax, cmap="Blues")
    eixo.set(xticks=indices, yticks=indices, xticklabels=base, yticklabels=base, title=titulo)
    eixo.tick_params(length=0, labelsize=10)
    for i, j in np.ndindex(matriz.shape):
        valor = matriz[i, j].real
        eixo.text(j, i, f"{valor:.3f}".replace(".", ","), ha="center", va="center",
                  color="white" if valor > 0.64 * vmax else "black", fontsize=9)
    return imagem


def visualizar_posicoes_conjuntas(rho, grade, sistemas, dims, *, titulo=None):
    """Retorna a figura e o eixo da densidade conjunta de posição de dois modos."""
    densidade = distribuicao_conjunta_posicoes(rho, grade, sistemas, dims)
    figura, eixo = plt.subplots(figsize=(6, 5), layout="constrained")
    imagem = eixo.contourf(grade, grade, densidade.T, levels=40, cmap="viridis")
    i, j = sorted(sistemas)
    eixo.set(aspect="equal", xlabel=rf"$x_{i}$", ylabel=rf"$x_{j}$", title=titulo)
    figura.colorbar(imagem, ax=eixo, label=rf"$P(x_{i}, x_{j})$")
    return figura, eixo


def visualizar_taxas_dual(epsilons, fatores, residuos, *, tolerancia, titulo):
    """Retorna figura e eixo com os resíduos finais por fator de aprendizado."""
    figura, eixo = plt.subplots(figsize=(9, 5), layout="constrained")
    for fator, erros in zip(fatores, residuos):
        eixo.loglog(epsilons, erros, marker="o", markersize=4, label=f"{fator:g}")
    eixo.axhline(tolerancia, color="0.4", linestyle="--", label="Tolerância")
    eixo.set(xlabel=r"Regularização $\epsilon$",
             ylabel=r"Norma do resíduo final $\|r\|_2$", title=titulo)
    eixo.grid(alpha=0.25, which="both")
    eixo.legend(title=r"Taxa $c\epsilon$", ncols=2)
    return figura, eixo
