#!/usr/bin/env python3
"""
Pre-download WhisperX models for serverless deployment.
Run this script during container build or deployment preparation.
"""

import logging
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import whisperx

# Constants
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2"]
ALIGNMENT_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"]
DEVICE = "cpu"  # Force CPU for serverless compatibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def preload_whisper_models() -> None:
    """Pre-download Whisper models for different sizes."""

    for model_size in WHISPER_MODELS:
        try:
            logger.info(f"Downloading Whisper model: {model_size}")
            model = whisperx.load_model(model_size, DEVICE, compute_type="int8")
            logger.info(f"✅ {model_size} model downloaded successfully")
            del model  # Free memory
        except Exception as e:
            logger.error(f"❌ Failed to download {model_size}: {e}")


def preload_alignment_models() -> None:
    """Pre-download alignment models for common languages."""
    for lang in ALIGNMENT_LANGUAGES:
        try:
            logger.info(f"Downloading alignment model for: {lang}")
            model_a, metadata = whisperx.load_align_model(
                language_code=lang, device=DEVICE
            )
            logger.info(f"✅ {lang} alignment model downloaded successfully")
            del model_a, metadata  # Free memory
        except Exception as e:
            logger.warning(f"⚠️ Could not download alignment for {lang}: {e}")


def preload_diarization_model() -> None:
    """Pre-download speaker diarization model."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("TOKEN")

    if not hf_token:
        logger.warning("⚠️ No HF_TOKEN found, skipping diarization model")
        return

    try:
        logger.info("Downloading diarization model...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=DEVICE
        )
        logger.info("✅ Diarization model downloaded successfully")
        del diarize_model  # Free memory
    except Exception as e:
        logger.error(f"❌ Failed to download diarization model: {e}")


def main() -> None:
    """Main preloading function."""
    logger.info("🚀 Starting model pre-loading for serverless deployment")

    # Check cache directory
    cache_dir = Path.home() / ".cache" / "whisperx"
    logger.info(f"Models will be cached in: {cache_dir}")

    # Download models
    preload_whisper_models()
    preload_alignment_models()
    preload_diarization_model()

    logger.info("✅ Model pre-loading complete!")
    logger.info(
        f"Cache size: {sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB"
    )


if __name__ == "__main__":
    main()
