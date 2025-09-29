import json
import logging
import os
import tempfile
import traceback
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set

import soundfile as sf
import torch
import whisperx
from pyannote.audio import Pipeline


class WhisperXTranscriber:
    SUPPORTED_FORMATS: Set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"}

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.logger = self._setup_logging()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and torch.cuda.is_available():
                self.logger.warning(
                    "CUDA is available but using CPU. Specify --device cuda for faster processing."
                )
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
            self.device = "cpu"

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Model size: {self.model_size}")

        self.model: Optional[Any] = None
        self.model_a: Optional[Any] = None
        self.metadata: Optional[Any] = None
        self.diarize_model: Optional[Pipeline] = None

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_models(self) -> None:
        if self.model is None:
            self.logger.info("Loading Whisper model...")
            try:
                self.model = whisperx.load_model(
                    self.model_size,
                    self.device,
                    compute_type="float16" if self.device == "cuda" else "int8",
                )
                self.logger.info("Whisper model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}")

    def _load_alignment_model(self, language_code: str) -> None:
        last_lang = getattr(self, "_last_alignment_language", None)
        if self.model_a is None or last_lang != language_code:
            self.logger.info(f"Loading alignment model for language: {language_code}")
            try:
                self.model_a, self.metadata = whisperx.load_align_model(
                    language_code=language_code, device=self.device
                )
                self._last_alignment_language = language_code
                self.logger.info("Alignment model loaded successfully")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load alignment model for {language_code}: {e}"
                )
                self.model_a, self.metadata = None, None

    def _load_diarization_model(self) -> None:
        if self.diarize_model is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("TOKEN")
            if not hf_token:
                raise ValueError(
                    "Hugging Face token required for speaker diarization. "
                    "Please set HF_TOKEN or TOKEN environment variable. "
                    "Get your token from: https://huggingface.co/settings/tokens"
                )

            self.logger.info("Loading diarization model...")
            try:
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
                )
                if self.diarize_model is not None:
                    self.diarize_model = self.diarize_model.to(
                        torch.device(self.device)
                    )
                self.logger.info("Diarization model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load diarization model: {e}")

    def validate_audio_file(self, audio_path: str) -> None:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

    def transcribe_audio(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.validate_audio_file(audio_path)
        self._load_models()

        self.logger.info("Transcribing audio...")
        try:
            audio = whisperx.load_audio(audio_path)
            if self.model is None:
                raise RuntimeError("Model not loaded")
            result = self.model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", "en")

            language_code = self.language or detected_language
            self.logger.info(
                f"{'Using specified' if self.language else 'Detected'} language: {language_code}"
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        self.logger.info("Performing word-level alignment...")
        try:
            self._load_alignment_model(language_code)
            if self.model_a is not None:
                result = whisperx.align(
                    result["segments"],
                    self.model_a,
                    self.metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
            else:
                self.logger.warning("Skipping alignment due to model loading failure")
        except Exception as e:
            self.logger.warning(f"Alignment failed, continuing without: {e}")

        self.logger.info("Performing speaker diarization...")
        try:
            self._load_diarization_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                sf.write(temp_wav.name, audio, 16000)
                temp_wav_path = temp_wav.name

            if self.diarize_model is None:
                raise RuntimeError("Diarization model not loaded")
            diarization_result = self.diarize_model(temp_wav_path)
            os.unlink(temp_wav_path)

            speaker_segments = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                speaker_segments.append(
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                )

            for segment in result.get("segments", []):
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)
                assigned_speaker = self._assign_speaker_to_segment(
                    segment_start, segment_end, speaker_segments
                )
                segment["speaker"] = assigned_speaker

        except Exception as e:
            self.logger.warning(f"Speaker diarization failed: {e}")
            self.logger.warning(f"Full traceback: {traceback.format_exc()}")
            for segment in result.get("segments", []):
                segment["speaker"] = "SPEAKER_00"

        audio_duration = len(audio) / 16000
        speakers_detected = len(
            set(seg.get("speaker", "SPEAKER_00") for seg in result.get("segments", []))
        )

        formatted_result = {
            "metadata": {
                "audio_file": os.path.basename(audio_path),
                "duration": round(audio_duration, 2),
                "model": self.model_size,
                "language": language_code,
                "speakers_detected": speakers_detected,
            },
            "segments": result.get("segments", []),
        }

        self.logger.info(
            f"Transcription complete. Found {len(formatted_result['segments'])} "
            f"segments with {speakers_detected} speakers."
        )
        return formatted_result

    def _assign_speaker_to_segment(
        self,
        segment_start: float,
        segment_end: float,
        speaker_segments: List[Dict[str, Any]],
    ) -> str:
        best_speaker = "SPEAKER_00"
        max_overlap = 0

        for spk_seg in speaker_segments:
            overlap_start = max(segment_start, spk_seg["start"])
            overlap_end = min(segment_end, spk_seg["end"])
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = spk_seg["speaker"]

        return best_speaker

    def format_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def format_output(
        self, transcription_result: Dict[str, Any], format_type: str = "txt"
    ) -> str:
        segments = transcription_result.get("segments", [])
        format_lower = format_type.lower()

        formatters = {
            "json": self._format_json,
            "txt": self._format_txt,
            "srt": self._format_srt,
            "vtt": self._format_vtt,
        }

        formatter = formatters.get(format_lower)
        if not formatter:
            supported = ", ".join(formatters.keys())
            raise ValueError(
                f"Unsupported format: {format_type}. Supported: {supported}"
            )

        return formatter(transcription_result, segments)

    def _format_json(
        self, transcription_result: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> str:
        return json.dumps(transcription_result, indent=2, ensure_ascii=False)

    def _format_txt(
        self, transcription_result: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> str:
        lines = []
        for segment in segments:
            start_time = self.format_timestamp(segment.get("start", 0))
            end_time = self.format_timestamp(segment.get("end", 0))
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment.get("text", "").strip()
            lines.append(f"[{start_time} --> {end_time}] {speaker}: {text}")
        return "\n".join(lines)

    def _format_srt(
        self, transcription_result: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> str:
        lines = []
        for i, segment in enumerate(segments, 1):
            start_time = self.format_timestamp(segment.get("start", 0)).replace(
                ".", ","
            )
            end_time = self.format_timestamp(segment.get("end", 0)).replace(".", ",")
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment.get("text", "").strip()
            lines.extend(
                [str(i), f"{start_time} --> {end_time}", f"{speaker}: {text}", ""]
            )
        return "\n".join(lines)

    def _format_vtt(
        self, transcription_result: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> str:
        lines = ["WEBVTT", ""]
        for segment in segments:
            start_time = self.format_timestamp(segment.get("start", 0))
            end_time = self.format_timestamp(segment.get("end", 0))
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment.get("text", "").strip()
            lines.extend([f"{start_time} --> {end_time}", f"{speaker}: {text}", ""])
        return "\n".join(lines)
