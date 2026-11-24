"""Compatibilidade temporária com os imports anteriores.

A implementação está em :mod:`modulos`; novos códigos devem importar de lá.
"""
from modulos import *  # noqa: F401,F403
from modulos import __all__
