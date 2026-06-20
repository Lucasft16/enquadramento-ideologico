"""Carrega e expõe a configuração global do projeto a partir de config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Carrega o arquivo YAML de configuração e retorna um dicionário.

    Args:
        path: Caminho alternativo para o arquivo de configuração.
              Se None, usa config.yaml na raiz do projeto.

    Returns:
        Dicionário com os parâmetros de configuração.
    """
    target = Path(path) if path else _CONFIG_PATH
    with target.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Instância compartilhada (somente leitura depois do carregamento).
CONFIG: dict[str, Any] = load_config()
