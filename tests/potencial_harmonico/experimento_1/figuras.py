"""Destino local das figuras dos slides."""

from modulos.visualizacao import salvar_figura

from .base import PASTA


def salvar(figura, nome):
    """Grava e fecha a figura e retorna o caminho do PNG."""
    return salvar_figura(figura, PASTA / "figuras_slides" / f"{nome}.png")
