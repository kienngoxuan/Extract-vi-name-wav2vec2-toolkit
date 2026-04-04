"""Pytest configuration and shared fixtures."""
import pytest


@pytest.fixture
def sample_vietnamese_text():
    """Sample Vietnamese text for testing."""
    return "Vũ Thị Yến"


@pytest.fixture
def sample_transcript():
    """Sample transcription output."""
    return "vũ thị yến"


@pytest.fixture
def sample_expected_name():
    """Sample expected name from filename."""
    return "vu_thi_yen"
