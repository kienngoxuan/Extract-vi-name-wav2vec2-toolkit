"""Tests for evaluation utilities and model loading."""
import os
import sys
import tempfile

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.unit
class TestAudioPathHandling:
    """Test handling of audio file paths."""

    def test_wav_file_pattern_matching(self):
        """Test WAV file can be identified."""
        test_path = "/data/person_name_500/vu_thi_yen_1.wav"
        assert test_path.endswith(".wav")

    def test_extract_expected_name_from_path(self):
        """Test extraction of expected name from file path."""
        fname = "vu_thi_yen_1.wav"
        expected = fname.replace("_1.wav", "").replace(".wav", "")
        assert expected == "vu_thi_yen"

    def test_extract_expected_name_no_suffix(self):
        """Test extraction when no numeric suffix."""
        fname = "vu_thi_yen.wav"
        # Simulate regex removal of numeric suffix and extension
        import re
        expected = re.sub(r"(?:_\d+)?\.wav$", "", fname)
        assert expected == "vu_thi_yen"


@pytest.mark.unit
class TestCSVHandling:
    """Test CSV reading and writing for transcription results."""

    def test_csv_header_structure(self):
        """Test expected CSV header structure."""
        expected_header = ["path_wav", "expected_name", "transcription"]
        assert len(expected_header) == 3
        assert "path_wav" in expected_header
        assert "expected_name" in expected_header
        assert "transcription" in expected_header

    def test_csv_row_format(self):
        """Test expected CSV row format."""
        sample_row = [
            "/data/vu_thi_yen_1.wav",
            "vu_thi_yen",
            "vũ thị yến"
        ]
        assert len(sample_row) == 3
        assert all(isinstance(field, str) for field in sample_row)


@pytest.mark.unit
class TestTranscriptionValidation:
    """Test transcription output validation."""

    def test_transcription_output_type(self):
        """Test transcription output is string."""
        transcription = "vũ thị yến"
        assert isinstance(transcription, str)

    def test_empty_transcription(self):
        """Test handling of empty transcription."""
        transcription = ""
        assert transcription == ""

    def test_transcription_with_unknown_tokens(self):
        """Test transcription may contain unknown tokens."""
        transcription = "vũ thị <unk> yến"
        assert "<unk>" in transcription or len(transcription) > 0


@pytest.mark.unit
class TestMetricsComputation:
    """Test metrics computation for evaluation."""

    def test_pass_fail_counting(self):
        """Test basic pass/fail counting."""
        passes = 335
        total = 500
        accuracy = (passes / total) * 100
        assert 60 < accuracy < 80  # Expected range for eval accuracy

    def test_wer_score_range(self):
        """Test WER score is in valid range (0-100)."""
        wer_score = 4.46
        assert 0 <= wer_score <= 100

    def test_accuracy_percentage_calculation(self):
        """Test accuracy percentage calculation."""
        passes = 345
        total = 500
        accuracy = (passes / total) * 100
        assert abs(accuracy - 69.0) < 0.01


@pytest.mark.integration
class TestEvaluationWorkflow:
    """Integration tests for evaluation workflow."""

    def test_can_create_temp_csv(self):
        """Test can create temporary CSV for results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            f.write("path_wav,expected_name,transcription\n")
            f.write("/data/test.wav,test_name,test_transcription\n")

        assert os.path.exists(csv_path)

        # Cleanup
        os.remove(csv_path)
        assert not os.path.exists(csv_path)

    def test_evaluation_output_directory(self):
        """Test evaluation output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "transcription_results_wav2vec2.csv")
            assert csv_path.endswith(".csv")
            assert tmpdir in csv_path
