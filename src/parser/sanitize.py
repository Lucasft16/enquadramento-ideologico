"""Limpeza e normalização de texto bruto antes do processamento."""

from __future__ import annotations

import re
import unicodedata


def sanitize(text: str) -> str:
    """Normaliza e limpa um texto bruto.

    Operações realizadas em ordem:
    1. Normalização Unicode NFC.
    2. Conversão para minúsculas.
    3. Remoção de URLs.
    4. Remoção de menções (@user) e hashtags (#tag).
    5. Substituição de hifens compostos e travessões por espaço.
    6. Remoção de pontuação (exceto apóstrofo interno).
    7. Colapso de espaços múltiplos.

    Args:
        text: Texto bruto a ser limpo.

    Returns:
        Texto normalizado como string única.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove menções e hashtags
    text = re.sub(r"[@#]\w+", " ", text)
    # Hifens compostos e travessões → espaço
    text = re.sub(r"[–—‒―]", " ", text)
    # Remove pontuação (mantém letras, dígitos, espaço e apóstrofo interno)
    text = re.sub(r"[^\w\s']", " ", text)
    # Apóstrofo não interno → espaço (ex: início/fim de token)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    # Underscores (de \w) → espaço
    text = re.sub(r"_", " ", text)
    # Colapsa espaços
    text = re.sub(r"\s+", " ", text).strip()
    return text
