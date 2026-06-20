"""Testes para o módulo parser (sanitize, stopwords, windows, pipeline)."""

import pytest

from src.parser.sanitize import sanitize
from src.parser.stopwords import remove_stopwords, get_stopwords
from src.parser.windows import sliding_windows, sentence_windows
from src.datastructures.trie import Trie


class TestSanitize:
    def test_lowercases(self):
        assert sanitize("Python É Bom") == "python é bom"

    def test_removes_url(self):
        result = sanitize("visite https://example.com agora")
        assert "http" not in result
        assert "example" not in result

    def test_removes_punctuation(self):
        result = sanitize("olá, mundo! tudo bem?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_spaces(self):
        result = sanitize("muito   espaço   aqui")
        assert "  " not in result

    def test_removes_hashtag_and_mention(self):
        result = sanitize("oi @fulano #politica hoje")
        assert "@" not in result
        assert "#" not in result

    def test_normalizes_unicode(self):
        # NFC: caracteres compostos devem ser preservados
        result = sanitize("ação")
        assert "ação" in result

    def test_replaces_em_dash(self):
        result = sanitize("estado—mínimo")
        assert "estado mínimo" in result or "estado" in result


class TestStopwords:
    def test_remove_common_stopwords(self):
        tokens = ["o", "estado", "é", "grande"]
        result = remove_stopwords(tokens)
        assert "o" not in result
        assert "é" not in result
        assert "estado" in result

    def test_removes_single_char(self):
        tokens = ["a", "b", "boa"]
        result = remove_stopwords(tokens)
        assert "a" not in result
        # "b" é stopword? Não, mas é 1 char — deve ser removido pelo len > 1
        assert "b" not in result
        assert "boa" in result

    def test_get_stopwords_returns_frozenset(self):
        sw = get_stopwords()
        assert isinstance(sw, frozenset)
        assert "de" in sw


class TestSlidingWindows:
    def test_basic_window(self):
        tokens = ["a", "b", "c", "d"]
        result = sliding_windows(tokens, 3)
        assert result == [["a", "b", "c"], ["b", "c", "d"]]

    def test_window_size_equals_len(self):
        tokens = ["x", "y", "z"]
        result = sliding_windows(tokens, 3)
        assert result == [["x", "y", "z"]]

    def test_window_size_larger_than_tokens(self):
        tokens = ["a", "b"]
        result = sliding_windows(tokens, 5)
        # Nenhuma janela completa → lista vazia
        assert result == []

    def test_short_tokens_returns_empty(self):
        # Menos de 2 tokens → sem coocorrências
        result = sliding_windows(["único"], 3)
        assert result == []

    def test_raises_on_invalid_window_size(self):
        with pytest.raises(ValueError):
            sliding_windows(["a", "b", "c"], 1)

    def test_sentence_windows_no_cross(self):
        sents = [["a", "b", "c"], ["x", "y", "z"]]
        result = sentence_windows(sents, 2)
        # Janelas dentro de cada sentença, sem cruzamento
        assert ["c", "x"] not in result
        assert len(result) == 4  # 2 janelas por sentença


class TestPipelineWithoutSpacy:
    """Testa o pipeline desligando o lematizador (sem dependência de spaCy)."""

    def test_process_document_returns_windows(self):
        from src.parser.pipeline import process_document

        text = "O mercado livre gera riqueza e prosperidade econômica para todos"
        windows = process_document(text, use_lemmatizer=False, window_size=3)
        # Deve retornar ao menos uma janela
        assert len(windows) > 0
        for w in windows:
            assert len(w) <= 3

    def test_process_document_removes_stopwords(self):
        from src.parser.pipeline import process_document

        text = "o e a de do para com em"
        windows = process_document(text, use_lemmatizer=False, window_size=3)
        # Todas as palavras são stopwords → nenhuma janela
        assert windows == []

    def test_process_document_applies_trie(self):
        from src.parser.pipeline import process_document

        trie = Trie()
        trie.insert("livre mercado")
        text = "livre mercado gera riqueza econômica real agora"
        windows = process_document(text, trie=trie, use_lemmatizer=False, window_size=3)
        # O marcador multipalavra deve aparecer como token único
        all_tokens = [tok for w in windows for tok in w]
        assert "livre_mercado" in all_tokens

    def test_process_document_empty_text(self):
        from src.parser.pipeline import process_document

        windows = process_document("", use_lemmatizer=False)
        assert windows == []
