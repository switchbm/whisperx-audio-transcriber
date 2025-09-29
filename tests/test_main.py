import argparse
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from main import (
    find_audio_files,
    get_output_filename,
    setup_output_directory,
    validate_arguments,
)


class TestArgumentValidation:
    """Test command line argument validation."""

    def test_validate_arguments_missing_input(self):
        """Test validation fails when no input specified."""
        args = argparse.Namespace(audio=None, batch=None)
        with pytest.raises(
            ValueError, match="Either --audio or --batch must be specified"
        ):
            validate_arguments(args)

    def test_validate_arguments_both_inputs(self):
        """Test validation fails when both inputs specified."""
        args = argparse.Namespace(audio="test.mp3", batch="test_dir")
        with pytest.raises(ValueError, match="Cannot specify both --audio and --batch"):
            validate_arguments(args)

    def test_validate_arguments_audio_not_exists(self):
        """Test validation fails when audio file doesn't exist."""
        args = argparse.Namespace(
            audio="nonexistent.mp3",
            batch=None,
            model="base",
            output_format="all",
            device="cpu",
        )
        with pytest.raises(FileNotFoundError):
            validate_arguments(args)

    def test_validate_arguments_invalid_model(self):
        """Test validation fails with invalid model."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            args = argparse.Namespace(
                audio=temp_path,
                batch=None,
                model="invalid",
                output_format="all",
                device="cpu",
            )
            with pytest.raises(ValueError, match="Invalid model"):
                validate_arguments(args)
        finally:
            os.unlink(temp_path)

    def test_validate_arguments_invalid_format(self):
        """Test validation fails with invalid output format."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            args = argparse.Namespace(
                audio=temp_path,
                batch=None,
                model="base",
                output_format="invalid",
                device="cpu",
            )
            with pytest.raises(ValueError, match="Invalid output format"):
                validate_arguments(args)
        finally:
            os.unlink(temp_path)

    def test_validate_arguments_valid(self):
        """Test validation passes with valid arguments."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            args = argparse.Namespace(
                audio=temp_path,
                batch=None,
                model="base",
                output_format="all",
                device="cpu",
            )
            validate_arguments(args)  # Should not raise
        finally:
            os.unlink(temp_path)


class TestFileOperations:
    """Test file and directory operations."""

    def test_find_audio_files_empty_directory(self):
        """Test finding audio files in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = find_audio_files(temp_dir)
            assert files == []

    def test_find_audio_files_with_audio(self):
        """Test finding audio files in directory with audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            audio_files = ["test1.mp3", "test2.wav", "test3.m4a"]
            other_files = ["readme.txt", "data.json"]

            for filename in audio_files + other_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, "w") as f:
                    f.write("test content")

            found_files = find_audio_files(temp_dir)

            # Should find only audio files, sorted
            expected_files = [os.path.join(temp_dir, f) for f in sorted(audio_files)]
            assert found_files == expected_files

    def test_find_audio_files_subdirectories(self):
        """Test finding audio files in subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectory structure
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)

            # Create audio files in both directories
            files = [
                os.path.join(temp_dir, "root.mp3"),
                os.path.join(subdir, "sub.wav"),
            ]

            for filepath in files:
                with open(filepath, "w") as f:
                    f.write("test content")

            found_files = find_audio_files(temp_dir)
            assert len(found_files) == 2
            assert all(f in found_files for f in files)

    def test_get_output_filename(self):
        """Test output filename generation."""
        filename = get_output_filename("/path/to/audio.mp3", "/output", "txt")
        assert filename == "/output/audio.txt"

        filename = get_output_filename("test.wav", ".", "json")
        assert filename == "./test.json"

    def test_setup_output_directory(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output", "subdir")
            result_path = setup_output_directory(output_dir)

            assert os.path.exists(output_dir)
            assert str(result_path) == output_dir


class TestMockIntegration:
    """Test integration with mocked dependencies."""

    @patch("main.WhisperXTranscriber")
    def test_process_single_file_success(self, mock_transcriber_class):
        """Test successful single file processing."""
        # Setup mocks
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_audio.return_value = {
            "metadata": {"duration": 60.0, "language": "en", "speakers_detected": 2},
            "segments": [{"text": "Hello world"}],
        }
        mock_transcriber.format_output.return_value = "Formatted output"
        mock_transcriber_class.return_value = mock_transcriber

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                from main import process_single_file

                args = argparse.Namespace(
                    output_dir=output_dir,
                    output_format="txt",
                    min_speakers=None,
                    max_speakers=None,
                    verbose=False,
                )

                # This should not raise an exception
                process_single_file(temp_path, mock_transcriber, args)

                # Verify transcriber was called
                mock_transcriber.transcribe_audio.assert_called_once()
                mock_transcriber.format_output.assert_called_once()

            finally:
                os.unlink(temp_path)
