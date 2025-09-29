#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import List

from tqdm import tqdm

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from transcriber import WhisperXTranscriber

# Constants
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
SUPPORTED_OUTPUT_FORMATS = ["txt", "json", "srt", "vtt", "all"]
SUPPORTED_DEVICES = ["cpu", "cuda"]


def timestamp_print(*args: Any, **kwargs: Any) -> None:
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}", *args, **kwargs)


def setup_output_directory(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_output_filename(audio_path: str, output_dir: str, format_type: str) -> str:
    audio_name = Path(audio_path).stem
    extension = "txt" if format_type == "txt" else format_type
    return os.path.join(output_dir, f"{audio_name}.{extension}")


def save_output(content: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def process_single_file(
    audio_path: str, transcriber: WhisperXTranscriber, args: argparse.Namespace
) -> None:
    try:
        timestamp_print(f"Processing: {audio_path}")

        result = transcriber.transcribe_audio(
            audio_path, min_speakers=args.min_speakers, max_speakers=args.max_speakers
        )

        output_dir = setup_output_directory(args.output_dir)
        formats = (
            ["txt", "json", "srt", "vtt"]
            if args.output_format == "all"
            else [args.output_format]
        )

        for format_type in formats:
            formatted_content = transcriber.format_output(result, format_type)
            output_path = get_output_filename(audio_path, str(output_dir), format_type)
            save_output(formatted_content, output_path)
            timestamp_print(f"Saved {format_type.upper()} output: {output_path}")

        metadata = result.get("metadata", {})
        timestamp_print("✅ Transcription complete!")
        timestamp_print(f"   Duration: {metadata.get('duration', 'unknown')}s")
        timestamp_print(f"   Language: {metadata.get('language', 'unknown')}")
        timestamp_print(f"   Speakers: {metadata.get('speakers_detected', 'unknown')}")
        timestamp_print(f"   Segments: {len(result.get('segments', []))}")

    except Exception as e:
        timestamp_print(f"❌ Error processing {audio_path}: {e}")
        if args.verbose:
            traceback.print_exc()


def find_audio_files(directory: str) -> List[str]:
    supported_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"}
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                audio_files.append(os.path.join(root, file))

    return sorted(audio_files)


def process_batch(
    batch_dir: str, transcriber: WhisperXTranscriber, args: argparse.Namespace
) -> None:
    if not os.path.isdir(batch_dir):
        raise ValueError(f"Batch directory not found: {batch_dir}")

    audio_files = find_audio_files(batch_dir)

    if not audio_files:
        timestamp_print(f"No supported audio files found in: {batch_dir}")
        timestamp_print(
            f"Supported formats: {', '.join(WhisperXTranscriber.SUPPORTED_FORMATS)}"
        )
        return

    timestamp_print(f"Found {len(audio_files)} audio files to process")

    successful = 0
    failed = 0

    for audio_file in tqdm(audio_files, desc="Processing files"):
        try:
            process_single_file(audio_file, transcriber, args)
            successful += 1
        except Exception as e:
            timestamp_print(f"❌ Failed to process {audio_file}: {e}")
            failed += 1
            if args.verbose:
                traceback.print_exc()

    timestamp_print("\n📊 Batch processing complete:")
    timestamp_print(f"   ✅ Successful: {successful}")
    timestamp_print(f"   ❌ Failed: {failed}")


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.audio and not args.batch:
        raise ValueError("Either --audio or --batch must be specified")

    if args.audio and args.batch:
        raise ValueError("Cannot specify both --audio and --batch")

    if args.audio and not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if args.batch and not os.path.isdir(args.batch):
        raise FileNotFoundError(f"Batch directory not found: {args.batch}")

    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model: {args.model}. Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    if args.output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format: {args.output_format}. Choose from: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )

    if args.device is not None and args.device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Invalid device: {args.device}. Choose from: {', '.join(SUPPORTED_DEVICES)}, or auto-detect (default)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WhisperX Audio Transcriber with Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe single file
  python main.py --audio meeting.mp3

  # Batch process directory
  python main.py --batch /path/to/audio/files/

  # Use specific model and language
  python main.py --audio interview.wav --model large-v2 --language en

  # Output specific format only
  python main.py --audio lecture.mp4 --output_format json

  # Specify speaker count hints
  python main.py --audio panel.mp3 --min_speakers 3 --max_speakers 5

Supported audio formats: mp3, wav, m4a, flac, ogg, mp4

Note: First run will download models (~2GB total). Requires HF_TOKEN environment
variable for speaker diarization. Get token from: https://huggingface.co/settings/tokens
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="Path to single audio file")
    input_group.add_argument(
        "--batch", type=str, help="Path to directory containing audio files"
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=SUPPORTED_MODELS,
        help="Whisper model size (default: base)",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help='Language code (e.g., "en", "es") or auto-detect if not specified',
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=SUPPORTED_DEVICES,
        help="Device to use (default: auto-detect)",
    )

    # Output options
    parser.add_argument(
        "--output_format",
        type=str,
        default="all",
        choices=SUPPORTED_OUTPUT_FORMATS,
        help="Output format (default: all)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )

    # Speaker diarization options
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=None,
        help="Minimum number of speakers for diarization",
    )

    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Maximum number of speakers for diarization",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and error details",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        # Validate arguments
        validate_arguments(args)

        # Check for HF_TOKEN or TOKEN
        if not (os.getenv("HF_TOKEN") or os.getenv("TOKEN")):
            timestamp_print(
                "⚠️  Warning: HF_TOKEN or TOKEN environment variable not set."
            )
            timestamp_print("   Speaker diarization requires a Hugging Face token.")
            timestamp_print(
                "   Get your token from: https://huggingface.co/settings/tokens"
            )
            timestamp_print("   Set it with: export HF_TOKEN=your_token_here")
            timestamp_print("   Or add TOKEN=your_token_here to your .env file")
            timestamp_print("   Continuing with transcription only...\n")

        # Initialize transcriber
        timestamp_print(f"Initializing WhisperX transcriber...")
        timestamp_print(f"Model: {args.model}")
        timestamp_print(f"Device: {args.device or 'auto-detect'}")
        timestamp_print(f"Language: {args.language or 'auto-detect'}")
        print()

        transcriber = WhisperXTranscriber(
            model_size=args.model, device=args.device, language=args.language
        )

        # Process files
        if args.batch:
            process_batch(args.batch, transcriber, args)
        else:
            process_single_file(args.audio, transcriber, args)

    except KeyboardInterrupt:
        timestamp_print("\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        timestamp_print(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
