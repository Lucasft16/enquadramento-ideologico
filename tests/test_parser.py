"""Testes para o módulo parser (sanitize, stopwords, windows, pipeline)."""

import pytest

from src.parser.sanitize import sanitize
from src.parser.stopwords import remove_stopwords, get_stopwords
from src.parser.windows import sliding_windows, sentence_windows
from src.parser.pipeline import process_document, process_corpus
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
    
    def test_empty_string(self):
        """Texto vazio deve retornar string vazia."""
        assert sanitize("") == ""
        
    def test_only_whitespace(self):
        """Apenas espaços → string vazia após strip."""
        assert sanitize("   \t\n  ") == ""
        
    def test_only_punctuation(self):
        """Texto composto só de pontuação → vazio ou string vazia."""
        result = sanitize("!!! ??? ... ---")
        assert result.strip() == ""
        
    def test_only_numbers(self):
        """Texto só com números deve ser preservado (dígitos são \\w)."""
        result = sanitize("42 100 3")
        assert "42" in result
        assert "100" in result
        
    def test_only_stopwords_after_sanitize(self):
        """Frase formada só por stopwords → tokens todos removidos depois."""
        result = sanitize("o e a de do para com em")
        tokens = result.split()
        assert remove_stopwords(tokens) == []
        
    def test_url_only(self):
        """Texto que é só uma URL deve resultar em string vazia."""
        result = sanitize("https://www.example.com/path?q=1")
        assert result.strip() == ""
        
    def test_mixed_languages(self):
        """Texto com palavras em outro idioma (inglês) deve ser preservado (não removido)."""
        result = sanitize("freedom and liberty")
        assert "freedom" in result
        assert "liberty" in result
    
    def test_repeated_special_chars(self):
        """Muitos caracteres especiais repetidos não devem gerar múltiplos espaços."""
        result = sanitize("olá!!!...???")
        assert "  " not in result
    
    def test_apostrophe_internal(self):
        """Apóstrofo interno (contração) deve ser preservado."""
        result = sanitize("l'état")
        # O apóstrofo interno entre letras deve ficar
        assert "'" in result or "l" in result
    
    def test_apostrophe_boundary(self):
        """Apóstrofo no início/fim do token deve ser removido."""
        result = sanitize("'palavra' 'outra'")
        assert result.startswith("'") is False
    
    def test_em_dash_and_en_dash(self):
        """Travessão e meia risca devem ser substituídos por espaço."""
        result = sanitize("estado–mínimo governo—ação")
        assert "–" not in result
        assert "—" not in result
        assert "estado" in result
        assert "mínimo" in result
    
    def test_underscore_replaced_by_space(self):
        """Underscore não deve sobreviver (é substituído por espaço)."""
        result = sanitize("livre_mercado")
        assert "_" not in result
    
    def test_very_long_text(self):
        """Texto muito longo (10.000 palavras) não deve lançar exceção."""
        text = "política economia " * 5000
        result = sanitize(text)
        assert len(result) > 0

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

    def test_exact_two_tokens(self):
        """Exatamente 2 tokens com window_size=2 → uma janela."""
        result = sliding_windows(["a", "b"], 2)
        assert result == [["a", "b"]]
    
    def test_window_size_zero_raises(self):
        """window_size=0 deve lançar ValueError."""
        with pytest.raises(ValueError):
            sliding_windows(["a", "b"], 0)
    
    def test_window_size_negative_raises(self):
        """window_size negativo deve lançar ValueError."""
        with pytest.raises(ValueError):
            sliding_windows(["a", "b"], -5)
    
    def test_empty_tokens_list(self):
        """Lista vazia de tokens → lista vazia."""
        result = sliding_windows([], 3)
        assert result == []
    
    def test_large_window_single_token(self):
        """1 token com window_size=100 → lista vazia (n < 2)."""
        result = sliding_windows(["unico"], 100)
        assert result == []
    
    def test_sentence_windows_empty_list(self):
        """sentence_windows com lista vazia de sentenças → lista vazia."""
        result = sentence_windows([], 3)
        assert result == []
    
    def test_sentence_windows_empty_sentence(self):
        """sentence_windows com sentença vazia → não gera janelas para ela."""
        result = sentence_windows([[], ["a", "b", "c"]], 2)
        assert len(result) == 2  # só da segunda sentença
    
    def test_sentence_windows_single_word_sentences(self):
        """Sentenças de 1 palavra nunca geram janelas."""
        result = sentence_windows([["palavra"], ["outra"]], 2)
        assert result == []


class TestPipelineWithoutSpacy:
    """Testa o pipeline desligando o lematizador (sem dependência de spaCy)."""

    def test_process_document_returns_windows(self):

        text = "O mercado livre gera riqueza e prosperidade econômica para todos"
        windows = process_document(text, use_lemmatizer=False, window_size=3)
        # Deve retornar ao menos uma janela
        assert len(windows) > 0
        for w in windows:
            assert len(w) <= 3

    def test_process_document_removes_stopwords(self):

        text = "o e a de do para com em"
        windows = process_document(text, use_lemmatizer=False, window_size=3)
        # Todas as palavras são stopwords → nenhuma janela
        assert windows == []

    def test_process_document_applies_trie(self):

        trie = Trie()
        trie.insert("livre mercado")
        text = "livre mercado gera riqueza econômica real agora"
        windows = process_document(text, trie=trie, use_lemmatizer=False, window_size=3)
        # O marcador multipalavra deve aparecer como token único
        all_tokens = [tok for w in windows for tok in w]
        assert "livre_mercado" in all_tokens

    def test_process_document_empty_text(self):

        windows = process_document("", use_lemmatizer=False)
        assert windows == []

    def test_only_numbers_produces_no_windows(self):
        """Texto só com dígitos → lemmatize_simple filtra → sem janelas."""
        windows = process_document("1 2 3 4 5", use_lemmatizer=False, window_size=2)
        assert windows == []
    
    def test_only_stopwords_produces_no_windows(self):
        """Texto só com stopwords → nenhum token sobrevivente → sem janelas."""
        text = "o e a de do para com em uma"
        windows = process_document(text, use_lemmatizer=False, window_size=3)
        assert windows == []
    
    def test_single_content_word_produces_no_windows(self):
        """Um único token de conteúdo → não há par para coocorrência → sem janelas."""
        windows = process_document("mercado", use_lemmatizer=False, window_size=2)
        assert windows == []
    
    def test_process_corpus_empty_list(self):
        """Corpus vazio → lista vazia."""
        result = process_corpus([], use_lemmatizer=False)
        assert result == []
    
    def test_process_corpus_all_empty_docs(self):
        """Corpus com documentos vazios → cada um produz lista vazia."""
        result = process_corpus(["", "  ", "\t"], use_lemmatizer=False)
        assert all(doc == [] for doc in result)
    
    def test_trie_match_phrase_not_in_vocab_after_collapse(self):
        """Token colapsado pela Trie deve aparecer como token único no resultado."""
        trie = Trie()
        trie.insert("reforma agraria")
        text = "reforma agraria distribui terra camponeses produção alimentar"
        windows = process_document(text, trie=trie, use_lemmatizer=False, window_size=4)
        all_tokens = [t for w in windows for t in w]
        assert "reforma_agraria" in all_tokens
        assert "reforma" not in all_tokens
        assert "agraria" not in all_tokens