"""
Serverless-optimized transcriber that assumes models are pre-loaded.
"""

import json
import logging
import os
import warnings
from datetime import timedelta
from typing import Any, Dict, List, Optional

import torch
import whisperx


class ServerlessTranscriber:
    """Serverless-optimized WhisperX transcriber."""

    # Global model cache (survives between lambda invocations)
    _model_cache: Dict[str, Any] = {}

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.device = "cpu"  # Force CPU for serverless
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup minimal logging for serverless."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
        return logger

    def _get_or_load_model(self, model_type: str, *args: Any, **kwargs: Any) -> Any:
        """Get model from cache or load if not cached."""
        cache_key = f"{model_type}_{hash((args, tuple(kwargs.items())))}"

        if cache_key not in self._model_cache:
            self.logger.info(f"Loading {model_type} (not in cache)")
            if model_type == "whisper":
                self._model_cache[cache_key] = whisperx.load_model(
                    self.model_size, self.device, compute_type="int8"
                )
            elif model_type == "alignment":
                self._model_cache[cache_key] = whisperx.load_align_model(
                    *args, device=self.device, **kwargs
                )
            elif model_type == "diarization":
                hf_token = os.getenv("HF_TOKEN") or os.getenv("TOKEN")
                self._model_cache[cache_key] = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=self.device
                )
        else:
            self.logger.info(f"{model_type} loaded from cache")

        return self._model_cache[cache_key]

    def transcribe_audio_fast(self, audio_path: str) -> Dict[str, Any]:
        """Fast transcription optimized for serverless."""
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)

            # Step 1: Transcribe
            model = self._get_or_load_model("whisper")
            result = model.transcribe(audio, batch_size=8)  # Smaller batch for memory
            language = result.get("language", "en")

            # Step 2: Alignment (skip if not critical)
            try:
                model_a, metadata = self._get_or_load_model("alignment", language)
                if model_a:
                    result = whisperx.align(
                        result["segments"], model_a, metadata, audio, self.device
                    )
            except Exception:
                self.logger.warning("Skipping alignment")

            # Step 3: Diarization (skip if token missing)
            try:
                if os.getenv("HF_TOKEN") or os.getenv("TOKEN"):
                    diarize_model = self._get_or_load_model("diarization")
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                else:
                    # Assign default speaker
                    for segment in result.get("segments", []):
                        segment["speaker"] = "SPEAKER_00"
            except Exception:
                self.logger.warning("Skipping diarization")
                for segment in result.get("segments", []):
                    segment["speaker"] = "SPEAKER_00"

            # Format result
            return {
                "metadata": {
                    "duration": len(audio) / 16000,
                    "language": language,
                    "model": self.model_size,
                    "speakers_detected": len(
                        set(
                            s.get("speaker", "SPEAKER_00")
                            for s in result.get("segments", [])
                        )
                    ),
                },
                "segments": result.get("segments", []),
            }

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise


# Global instance (reused across lambda invocations)
_transcriber_instance: Optional[ServerlessTranscriber] = None


def get_transcriber(model_size: str = "base") -> ServerlessTranscriber:
    """Get singleton transcriber instance."""
    global _transcriber_instance
    if _transcriber_instance is None or _transcriber_instance.model_size != model_size:
        _transcriber_instance = ServerlessTranscriber(model_size)
    return _transcriber_instance
