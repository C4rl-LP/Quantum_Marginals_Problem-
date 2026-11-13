"""API pública do QMP com carregamento sob demanda por submódulo.

Exemplos: ``from modulos import partial_trace`` ou
``from modulos.sdp import solve_qmp``. Dependências do solver são carregadas
somente quando um símbolo do módulo ``sdp`` é solicitado.
"""
from importlib import import_module

_EXPORTS = {
    "basis": "estados", "ketbra": "estados", "tensor": "estados",
    "ghz": "estados", "mestura": "estados", "zero": "estados", "one": "estados",
    "verifica_densidade": "validacao", "_verify_density_matrix": "validacao",
    "verify_marginals": "validacao",
    "partial_trace": "operadores", "base_hermitiana": "operadores",
    "embed_operator": "operadores", "parse_subsystem_key": "operadores",
    "build_qmp_sdp": "sdp", "solve_qmp": "sdp",
    "DualEntropico": "dual_entropico",
    "otimizar_dual": "otimizacao",
    "RestricoesMarginais": "marginais",
    "construir_restricoes_marginais": "marginais",
    "random_density_matrix": "experimentos", "build_product_marginals": "experimentos",
    "build_correlated_marginals": "experimentos",
    "check_overlap_consistency": "experimentos", "print_experiment_stats": "experimentos",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Importa o submódulo correspondente apenas quando o nome é solicitado."""
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__():
    """Inclui os nomes públicos carregados sob demanda na introspecção."""
    return sorted(set(globals()) | set(__all__))
