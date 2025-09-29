# 🎙️ WhisperX Audio Transcriber

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful, production-ready audio transcription tool built on WhisperX that provides **state-of-the-art speech-to-text** with **speaker diarization** (speaker identification). Perfect for transcribing meetings, interviews, podcasts, lectures, and any multi-speaker audio content.

## ✨ Features

- 🚀 **High-accuracy transcription** using OpenAI's Whisper models
- 👥 **Speaker diarization** to identify who is speaking when
- 📄 **Multiple output formats**: TXT, JSON, SRT, VTT
- ⚡ **Batch processing** for multiple files
- 🖥️ **GPU acceleration** with automatic CPU fallback
- ⏱️ **Word-level timestamps** for precise alignment
- 🌍 **Multi-language support** with auto-detection
- 📊 **Progress tracking** with timestamps and progress bars
- 🛡️ **Robust error handling** and validation

## 🎯 Demo

```bash
# Single command to transcribe with speaker identification
python main.py --audio meeting.mp3

# Output:
[2025-09-29 12:44:39] Processing: meeting.mp3
[2025-09-29 12:44:39] ✅ Transcription complete!
[2025-09-29 12:44:39]    Duration: 47.59s
[2025-09-29 12:44:39]    Language: en
[2025-09-29 12:44:39]    Speakers: 2
[2025-09-29 12:44:39]    Segments: 20
```

**Result:**
```
[00:00:00.622 --> 00:00:01.563] SPEAKER_00: Hello, how's it going?
[00:00:02.524 --> 00:00:03.406] SPEAKER_01: Good, how are you?
[00:00:04.267 --> 00:00:06.270] SPEAKER_00: Do you want to talk about your insurance policy?
[00:00:06.530 --> 00:00:07.613] SPEAKER_01: Yes, I would like that.
```

## 🎵 Supported Audio Formats

- **MP3** (.mp3) - Most common format
- **WAV** (.wav) - Uncompressed audio
- **M4A** (.m4a) - Apple/iTunes format
- **FLAC** (.flac) - Lossless compression
- **OGG** (.ogg) - Open source format
- **MP4** (.mp4) - Video files (audio extracted)

## 🚀 Quick Start

### 1. Install Dependencies

**macOS:**
```bash
# Install FFmpeg
brew install ffmpeg

# Clone repository
git clone https://github.com/yourusername/whisperx-audio-transcriber.git
cd whisperx-audio-transcriber

# Install Python dependencies
pip install -e .
```

**Ubuntu/Debian:**
```bash
# Install FFmpeg
sudo apt update && sudo apt install ffmpeg

# Clone and install
git clone https://github.com/yourusername/whisperx-audio-transcriber.git
cd whisperx-audio-transcriber
pip install -e .
```

### 2. Set Up Hugging Face Token (for speaker diarization)

**🔑 Speaker diarization requires accepting Pyannote model terms**

1. **Create Hugging Face account**: [huggingface.co](https://huggingface.co/)
2. **Generate access token**:
   - Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New Token" → Choose "Read" access → Copy token
3. **⚠️ IMPORTANT: Accept model terms** (must be done while logged in):
   - Visit [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) → Click "Agree and access repository"
   - Visit [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) → Click "Agree and access repository"
4. **Configure token**:

```bash
# Option 1: Environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: .env file (recommended)
echo "TOKEN=hf_your_token_here" > .env
```

**✅ Verification**: On first run, you should see "Diarization model loaded successfully"

### 3. Run Your First Transcription

```bash
# Single file
python main.py --audio your_audio.mp3

# Batch processing
python main.py --batch /path/to/audio/folder/

# Advanced options
python main.py --audio meeting.wav --model large-v2 --min_speakers 2 --max_speakers 5
```

## 📊 Model Performance Comparison

| Model | Size | Speed | Accuracy | Memory | Best For |
|-------|------|-------|----------|---------|-----------|
| `tiny` | 39MB | 🚀🚀🚀 | ⭐⭐⭐ | 1GB | Quick drafts |
| `base` | 74MB | 🚀🚀 | ⭐⭐⭐⭐ | 1GB | **Recommended** |
| `small` | 244MB | 🚀 | ⭐⭐⭐⭐ | 2GB | Better accuracy |
| `medium` | 769MB | 🐌 | ⭐⭐⭐⭐⭐ | 5GB | High accuracy |
| `large-v3` | 1.5GB | 🐌🐌 | ⭐⭐⭐⭐⭐ | 10GB | Maximum quality |

*Speed relative to real-time on GPU*

## 🔧 Advanced Usage

### Command Line Options

```bash
python main.py [OPTIONS]

Required (choose one):
  --audio FILE          Single audio file
  --batch DIRECTORY     Process all audio files in directory

Model Options:
  --model SIZE          tiny, base, small, medium, large-v2, large-v3 (default: base)
  --language CODE       Language code (en, es, fr, etc.) or auto-detect
  --device DEVICE       cpu, cuda, or auto-detect (default: auto)

Output Options:
  --output_format FORMAT  txt, json, srt, vtt, all (default: all)
  --output_dir DIR        Output directory (default: output)

Speaker Options:
  --min_speakers N      Minimum speakers for diarization
  --max_speakers N      Maximum speakers for diarization

Other:
  --verbose            Detailed logging and error traces
```

### Real-World Examples

**📞 Customer Service Call:**
```bash
python main.py --audio customer_call.mp3 --min_speakers 2 --max_speakers 2 --output_format json
```

**🎤 Podcast Episode:**
```bash
python main.py --audio podcast_ep1.mp3 --model medium --language en --output_format srt
```

**👥 Team Meeting:**
```bash
python main.py --audio standup.wav --min_speakers 4 --max_speakers 8 --verbose
```

**📚 Lecture Series (Batch):**
```bash
python main.py --batch ./lectures/ --model small --output_dir ./transcripts/
```

## 📁 Output Formats

### 📝 TXT - Human Readable
```
[00:00:00.000 --> 00:00:05.240] SPEAKER_00: Welcome to today's meeting.
[00:00:05.580 --> 00:00:08.920] SPEAKER_01: Thanks for having me here.
```

### 🔗 JSON - Structured Data
```json
{
  "metadata": {
    "audio_file": "meeting.mp3",
    "duration": 1847.2,
    "language": "en",
    "speakers_detected": 3,
    "model": "base"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 5.24,
      "speaker": "SPEAKER_00",
      "text": "Welcome to today's meeting."
    }
  ]
}
```

### 🎬 SRT - Subtitles
```srt
1
00:00:00,000 --> 00:00:05,240
SPEAKER_00: Welcome to today's meeting.

2
00:00:05,580 --> 00:00:08,920
SPEAKER_01: Thanks for having me here.
```

### 🌐 VTT - Web Subtitles
```vtt
WEBVTT

00:00:00.000 --> 00:00:05.240
SPEAKER_00: Welcome to today's meeting.

00:00:05.580 --> 00:00:08.920
SPEAKER_01: Thanks for having me here.
```

## ⚡ Performance Tips

### 🖥️ Hardware Optimization
- **GPU**: 5-10x faster than CPU
- **Memory**: 8GB+ RAM recommended for large models
- **Storage**: SSD preferred for large batch jobs

### 🎛️ Settings Optimization
```bash
# Fast processing (good quality)
python main.py --audio file.mp3 --model base --device cuda

# Maximum quality (slower)
python main.py --audio file.mp3 --model large-v3 --language en

# Batch optimization
python main.py --batch ./files/ --model small --output_format txt
```

## 🛠️ Troubleshooting

### Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| 🔒 Permission denied with uv | `export TMPDIR=/tmp` |
| 🚫 FFmpeg not found | Install FFmpeg: `brew install ffmpeg` |
| ⚠️ CUDA out of memory | Use `--model small` or `--device cpu` |
| 👥 Poor speaker separation | Add `--min_speakers 2 --max_speakers 4` |
| 🐌 Slow processing | Use smaller model or enable GPU |
| 🔑 HF_TOKEN missing | Set up Hugging Face token (see setup) |

### 📊 First Run Information
- **Download time**: 2-5 minutes (depending on internet)
- **Model size**: ~2GB total for all components
- **Cache location**: `~/.cache/whisperx/`
- **Subsequent runs**: Much faster (models cached)

## 🏗️ Development

### 🧪 Running Tests
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ --cov=.

# Code formatting
black . && isort . && flake8 .
```

### 🚀 Serverless Deployment
See `preload_models.py` and `serverless_transcriber.py` for optimized deployment patterns.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. ✍️ Make your changes with tests
4. ✅ Ensure tests pass (`pytest tests/`)
5. 📝 Update documentation if needed
6. 🚀 Submit a pull request

### 💡 Ideas for Contributions
- 🌍 Additional language support
- 📱 Web interface
- 🎨 GUI application
- 📊 Better visualization
- 🐳 Docker containerization
- ☁️ Cloud deployment guides

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### ⚖️ Important Notes
- **Code**: MIT License (free for commercial use)
- **Models**: pyannote.audio models require separate licensing for commercial use
- **Research use**: Completely free
- **Commercial use**: Contact pyannote team for model licensing

## 🙏 Acknowledgments

This project builds upon incredible work from:

- 🎯 [WhisperX](https://github.com/m-bain/whisperX) - Core transcription engine
- 🎤 [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition models
- 👥 [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- ⚡ [PyTorch](https://pytorch.org/) - Machine learning framework

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/whisperx-audio-transcriber&type=Date)](https://star-history.com/#yourusername/whisperx-audio-transcriber&Date)

---

<div align="center">

**Made with ❤️ by [Your Name]**

[⭐ Star this repo](https://github.com/yourusername/whisperx-audio-transcriber) • [🐛 Report Bug](https://github.com/yourusername/whisperx-audio-transcriber/issues) • [💡 Request Feature](https://github.com/yourusername/whisperx-audio-transcriber/issues)

</div>