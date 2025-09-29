import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from transcriber import WhisperXTranscriber


class TestWhisperXTranscriber:
    """Test cases for WhisperXTranscriber class."""

    def test_init_default_params(self):
        """Test transcriber initialization with default parameters."""
        transcriber = WhisperXTranscriber()
        assert transcriber.model_size == "base"
        assert transcriber.language is None
        assert transcriber.device in ["cuda", "cpu"]

    def test_init_custom_params(self):
        """Test transcriber initialization with custom parameters."""
        transcriber = WhisperXTranscriber(
            model_size="large-v2", device="cpu", language="en"
        )
        assert transcriber.model_size == "large-v2"
        assert transcriber.device == "cpu"
        assert transcriber.language == "en"

    def test_validate_audio_file_exists(self):
        """Test audio file validation with existing file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            transcriber = WhisperXTranscriber()
            transcriber.validate_audio_file(temp_path)  # Should not raise
        finally:
            os.unlink(temp_path)

    def test_validate_audio_file_not_exists(self):
        """Test audio file validation with non-existent file."""
        transcriber = WhisperXTranscriber()
        with pytest.raises(FileNotFoundError):
            transcriber.validate_audio_file("nonexistent.mp3")

    def test_validate_audio_file_unsupported_format(self):
        """Test audio file validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            transcriber = WhisperXTranscriber()
            with pytest.raises(ValueError, match="Unsupported audio format"):
                transcriber.validate_audio_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        transcriber = WhisperXTranscriber()

        # Test various timestamp values
        assert transcriber.format_timestamp(0) == "00:00:00.000"
        assert transcriber.format_timestamp(65.5) == "00:01:05.500"
        assert transcriber.format_timestamp(3725.123) == "01:02:05.123"

    def test_format_output_txt(self):
        """Test TXT format output."""
        transcriber = WhisperXTranscriber()

        result = {
            "metadata": {"audio_file": "test.mp3"},
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "speaker": "SPEAKER_00",
                    "text": "Hello world",
                },
                {
                    "start": 5.5,
                    "end": 10.0,
                    "speaker": "SPEAKER_01",
                    "text": "How are you?",
                },
            ],
        }

        output = transcriber.format_output(result, "txt")
        lines = output.split("\n")

        assert len(lines) == 2
        assert "SPEAKER_00: Hello world" in lines[0]
        assert "SPEAKER_01: How are you?" in lines[1]
        assert "00:00:00.000 --> 00:00:05.000" in lines[0]

    def test_format_output_json(self):
        """Test JSON format output."""
        transcriber = WhisperXTranscriber()

        result = {
            "metadata": {"audio_file": "test.mp3"},
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"}
            ],
        }

        output = transcriber.format_output(result, "json")
        parsed = json.loads(output)

        assert parsed["metadata"]["audio_file"] == "test.mp3"
        assert len(parsed["segments"]) == 1
        assert parsed["segments"][0]["text"] == "Hello"

    def test_format_output_srt(self):
        """Test SRT format output."""
        transcriber = WhisperXTranscriber()

        result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"}
            ]
        }

        output = transcriber.format_output(result, "srt")
        lines = output.split("\n")

        assert "1" in lines[0]  # Subtitle number
        assert "00:00:00,000 --> 00:00:05,000" in lines[1]  # Timestamp with commas
        assert "SPEAKER_00: Hello" in lines[2]  # Speaker and text

    def test_format_output_vtt(self):
        """Test VTT format output."""
        transcriber = WhisperXTranscriber()

        result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"}
            ]
        }

        output = transcriber.format_output(result, "vtt")
        lines = output.split("\n")

        assert "WEBVTT" in lines[0]
        assert "00:00:00.000 --> 00:00:05.000" in output
        assert "SPEAKER_00: Hello" in output

    def test_format_output_invalid_format(self):
        """Test invalid format raises error."""
        transcriber = WhisperXTranscriber()

        result = {"segments": []}

        with pytest.raises(ValueError, match="Unsupported format"):
            transcriber.format_output(result, "invalid")

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    @patch("whisperx.load_model")
    @patch("whisperx.load_audio")
    @patch("whisperx.load_align_model")
    @patch("whisperx.DiarizationPipeline")
    @patch("whisperx.align")
    @patch("whisperx.assign_word_speakers")
    def test_transcribe_audio_success(
        self,
        mock_assign,
        mock_align,
        mock_diarize,
        mock_align_model,
        mock_load_audio,
        mock_load_model,
    ):
        """Test successful audio transcription."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"start": 0, "end": 5, "text": "Hello"}],
        }
        mock_load_model.return_value = mock_model

        mock_load_audio.return_value = [0.1, 0.2, 0.3] * 16000  # 3 seconds of audio

        mock_align_model.return_value = (MagicMock(), MagicMock())
        mock_align.return_value = {
            "segments": [{"start": 0, "end": 5, "text": "Hello"}]
        }

        mock_diarize_instance = MagicMock()
        mock_diarize_instance.return_value = MagicMock()
        mock_diarize.return_value = mock_diarize_instance

        mock_assign.return_value = {
            "segments": [
                {"start": 0, "end": 5, "text": "Hello", "speaker": "SPEAKER_00"}
            ]
        }

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            transcriber = WhisperXTranscriber()
            result = transcriber.transcribe_audio(temp_path)

            # Verify result structure
            assert "metadata" in result
            assert "segments" in result
            assert result["metadata"]["language"] == "en"
            assert len(result["segments"]) == 1
            assert result["segments"][0]["speaker"] == "SPEAKER_00"

        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests that don't require actual models."""

    def test_supported_formats_list(self):
        """Test that supported formats are correctly defined."""
        expected_formats = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"}
        assert WhisperXTranscriber.SUPPORTED_FORMATS == expected_formats
