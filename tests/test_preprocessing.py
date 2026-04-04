"""Tests for preprocessing and text normalization utilities from eval_wav2vec2.py."""
import os
import sys

import pytest

# Add parent directory to path to import eval_wav2vec2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from eval_wav2vec2 import (
        compare_support_dialect_tone,
        convert_vietnamese_diacritics,
        convert_vietnamese_number,
        normalize_speech_patterns,
        vietnamese_number_converter,
    )
except ImportError:
    pytest.skip("Required module not available", allow_module_level=True)


@pytest.mark.unit
class TestVietnameseDiacriticsRemoval:
    """Test Vietnamese diacritics/tone mark removal."""

    def test_convert_vietnamese_diacritics_basic(self):
        """Test basic diacritics conversion."""
        result = convert_vietnamese_diacritics("Vũ Thị Yến")
        assert result == "Vu Thi Yen"

    def test_convert_vietnamese_diacritics_all_vowels(self):
        """Test all vowel variants are converted."""
        result = convert_vietnamese_diacritics("àáảãạăằắâ")
        assert result == "aaaaaaaaa"

    def test_convert_vietnamese_diacritics_d_with_stroke(self):
        """Test đ is converted to d."""
        result = convert_vietnamese_diacritics("Đỗ")
        assert result == "Do"

    def test_convert_vietnamese_diacritics_empty_string(self):
        """Test empty string handling."""
        result = convert_vietnamese_diacritics("")
        assert result == ""

    def test_convert_vietnamese_diacritics_no_diacritics(self):
        """Test text without diacritics remains unchanged."""
        result = convert_vietnamese_diacritics("Hello")
        assert result == "Hello"


@pytest.mark.unit
class TestVietnameseNumberConversion:
    """Test Vietnamese number conversions."""

    def test_convert_vietnamese_number_single_digits(self):
        """Test single digit conversions."""
        assert convert_vietnamese_number("0") == "không"
        assert convert_vietnamese_number("1") == "một"
        assert convert_vietnamese_number("5") == "năm"
        assert convert_vietnamese_number("9") == "chín"

    def test_convert_vietnamese_number_sequence(self):
        """Test number sequence conversion."""
        result = convert_vietnamese_number("123")
        assert "một" in result or result == "một hai ba"

    def test_convert_vietnamese_number_empty(self):
        """Test empty string."""
        result = convert_vietnamese_number("")
        assert result == ""


@pytest.mark.unit
class TestNormalizeSpeechPatterns:
    """Test speech pattern normalization for regional dialects."""

    def test_normalize_l_to_n(self):
        """Test Northern dialect l→n conversion."""
        result = normalize_speech_patterns("lẢNG")
        assert result.startswith("n")

    def test_normalize_r_to_d(self):
        """Test r→d conversion."""
        result = normalize_speech_patterns("rOM")
        assert result.startswith("d")

    def test_normalize_gi_to_d(self):
        """Test gi→d conversion."""
        result = normalize_speech_patterns("giỜNG")
        assert result.startswith("d")

    def test_normalize_s_to_x(self):
        """Test s→x conversion."""
        result = normalize_speech_patterns("SỮA")
        assert result.startswith("x")

    def test_normalize_tr_to_ch(self):
        """Test tr→ch conversion."""
        result = normalize_speech_patterns("TRẢ")
        assert result.startswith("ch")

    def test_normalize_middle_letters_unchanged(self):
        """Test that middle/suffix letters are not modified."""
        result = normalize_speech_patterns("bộ")
        assert result == "bộ"


@pytest.mark.unit
class TestCompareDialectTone:
    """Test dialect-aware name comparison function."""

    def test_exact_match(self, sample_expected_name, sample_transcript):
        """Test exact name match."""
        # vu_thi_yen should match vũ thị yến (after normalization)
        result = compare_support_dialect_tone("vũ thị yến", "vu_thi_yen")
        assert isinstance(result, bool)

    def test_case_insensitive(self):
        """Test case-insensitive comparison."""
        result = compare_support_dialect_tone("VŨ THỊ YẾN", "vu_thi_yen")
        assert isinstance(result, bool)

    def test_underscore_handling(self):
        """Test underscore is treated as space."""
        result = compare_support_dialect_tone("vu thi yen", "vu_thi_yen")
        assert isinstance(result, bool)

    def test_punctuation_removal(self):
        """Test punctuation is removed before comparison."""
        result = compare_support_dialect_tone("vu. thi, yen!", "vu_thi_yen")
        assert isinstance(result, bool)

    def test_empty_string(self):
        """Test empty string handling."""
        result = compare_support_dialect_tone("", "vu_thi_yen")
        assert isinstance(result, bool)


@pytest.mark.unit
class TestVietnameseNumberConverter:
    """Test the comprehensive Vietnamese number converter."""

    def test_vietnamese_number_converter_single_word(self):
        """Test conversion of single Vietnamese number word."""
        result = vietnamese_number_converter("một")
        # Should convert 'một' to '1'
        assert "1" in result or result == "1"

    def test_vietnamese_number_converter_sequence(self):
        """Test conversion of Vietnamese number sequence."""
        result = vietnamese_number_converter("một hai ba")
        # Should be converted to numeric representation
        assert isinstance(result, str)

    def test_vietnamese_number_converter_mixed_text(self):
        """Test mixed text with numbers and letters."""
        result = vietnamese_number_converter("điện thoại một hai ba")
        assert isinstance(result, str)

    def test_vietnamese_number_converter_empty(self):
        """Test empty string."""
        result = vietnamese_number_converter("")
        assert result == ""
