"""Destino e base computacional das figuras Beamer de qubits."""

from pathlib import Path

from modulos.visualizacao import salvar_figura, visualizar_matriz_densidade

DESTINO = Path(__file__).resolve().parent / "figuras_slides"


def salvar(figura, nome, dpi=200):
    """Grava o PNG, fecha a figura e retorna o caminho salvo."""
    return salvar_figura(figura, DESTINO / f"{nome}.png", dpi=dpi)


def matriz_densidade(eixo, matriz, titulo):
    """Mostra a parte real da matriz na base computacional de dois qubits."""
    return visualizar_matriz_densidade(eixo, matriz, titulo, base=("00", "01", "10", "11"))
