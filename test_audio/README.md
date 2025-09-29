# Test Audio Files

This directory contains sample audio files for testing the transcription functionality.

## Sample Files

Due to the large size of audio files, sample files are not included in the repository. To test the application, you can:

### Option 1: Create Test Audio Files

Use a text-to-speech tool to generate test audio:

```bash
# On macOS, use the built-in say command
say "Hello everyone, welcome to today's meeting. My name is John and I'll be leading the discussion." -o test_meeting_speaker1.wav

say "Thank you John. I'm Sarah, and I'm excited to participate in this meeting today." -o test_meeting_speaker2.wav

# Combine them with ffmpeg
ffmpeg -i test_meeting_speaker1.wav -i test_meeting_speaker2.wav -filter_complex "[0:0][1:0]concat=n=2:v=0:a=1[out]" -map "[out]" test_meeting_combined.wav
```

### Option 2: Download Sample Files

You can download free sample audio files from:

- [Freesound.org](https://freesound.org/) - Creative Commons audio files
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/) - Free BBC sound effects
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) - Open source voice data

### Option 3: Record Your Own

Record a short conversation using your phone or computer:

1. Record 30-60 seconds of conversation between 2-3 people
2. Save as MP3, WAV, or M4A format
3. Place in this directory
4. Test with: `python main.py --audio test_audio/your_recording.mp3`

## Recommended Test Scenarios

1. **Single Speaker** (30s): One person speaking clearly
2. **Two Speakers** (1-2 min): Conversation between two people
3. **Multiple Speakers** (2-3 min): Meeting with 3-5 participants
4. **Different Languages**: Non-English content to test language detection
5. **Poor Audio Quality**: Test robustness with background noise

## Expected Results

After transcription, you should see:

- Accurate text transcription
- Speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- Proper timestamps
- Multiple output formats (TXT, JSON, SRT, VTT)

Files will be saved to the `output/` directory by default.