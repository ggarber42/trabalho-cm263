"""Modelos do estudo (S5): interface comum em base.py; um arquivo por família."""
from src.models.base import REGISTRO, Modelo, lista_modelos_disponiveis  # noqa: F401
from src.models import series, paciente, extras, estatisticos  # noqa: F401  (registram os modelos)
