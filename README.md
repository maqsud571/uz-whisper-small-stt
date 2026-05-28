# Uzbek Whisper Small STT

`maqsudxo1ja/uz-whisper-small-stt` is a Whisper Small based Speech-to-Text model
prepared for converting Uzbek speech into text.

The model works with the Hugging Face `transformers` library and is configured
for Uzbek (`uz`) transcription by default. Base model:
[`openai/whisper-small`](https://huggingface.co/openai/whisper-small).

## Artifacts

- **Model:** https://huggingface.co/maqsudxo1ja/uz-whisper-small-stt
- **Task:** Automatic Speech Recognition (ASR), converting speech to text
- **Default language:** Uzbek (`uz`)
- **Default task:** `transcribe`
- **Model format:** `safetensors`
- **Library:** `transformers`
- **License:** Apache-2.0

## Model Overview

This model is an encoder-decoder model based on the `small` variant of the
Whisper architecture. The input audio signal is first converted to a 16 kHz
format, then transformed into log-Mel spectrogram features. The encoder encodes
the audio features, and the decoder converts them into text tokens.

The model configuration sets Uzbek language and transcription mode by default:

- `language`: `uz`
- `task`: `transcribe`
- `return_timestamps`: `false`
- `sampling_rate`: `16000`
- `chunk_length`: `30`
- `n_samples`: `480000`

## How It Works by Seconds

Whisper models typically process audio using a 30-second context window.
According to the `preprocessor_config.json` values in this repository:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `sampling_rate` | 16000 Hz | Audio is represented as 16000 samples per second |
| `chunk_length` | 30 seconds | Main audio window used by the model |
| `n_samples` | 480000 | 30 seconds x 16000 samples |
| `feature_size` | 80 | Number of log-Mel features |
| `hop_length` | 160 | Feature step of approximately every 10 ms |

For short audio files, such as 5-10 second clips, the model processes the given
audio segment and returns a single text result. Audio longer than 30 seconds
should be split into chunks. In practical usage, it is recommended to use
`chunk_length_s=30` and `stride_length_s=5` to reduce word loss around chunk
boundaries.

Inference time depends on the audio length, CPU/GPU type, RAM/VRAM availability,
and batch settings. **Real-Time Factor (RTF)** is used to measure speed:

```text
RTF = inference_seconds / audio_seconds
```

- `RTF < 1.0` - the model is faster than real time.
- `RTF = 1.0` - a 30-second audio file is processed in about 30 seconds.
- `RTF > 1.0` - processing is slower than the audio duration.

## Installation

```bash
pip install -r requirements.txt
```

Or install the dependencies separately:

```bash
pip install "transformers>=4.45.2" torch torchaudio safetensors
```

## Basic Usage

```python
from transformers import pipeline
import torch

model_id = "maqsudxo1ja/uz-whisper-small-stt"
device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    device=device,
)

result = asr("audio.wav")
print(result["text"])
```

## Direct Usage with `WhisperProcessor`

```python
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = "maqsudxo1ja/uz-whisper-small-stt"
audio_path = "audio.wav"
target_sr = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
model.eval()

waveform, sample_rate = torchaudio.load(audio_path)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

if sample_rate != target_sr:
    resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
    waveform = resampler(waveform)

inputs = processor(
    waveform.squeeze().numpy(),
    sampling_rate=target_sr,
    return_tensors="pt",
)

with torch.inference_mode():
    predicted_ids = model.generate(inputs.input_features.to(device))

text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(text.strip())
```

## Expected Audio Format

- Recommended sample rate: 16 kHz
- Channel: mono is preferred
- Formats: `wav`, `mp3`, `flac`, and other audio formats supported by
  `torchaudio` or the selected backend
- Clean audio recorded close to the microphone usually produces better results

## Evaluation

This model card does not include verified WER/CER results yet. For a professional
benchmark, evaluate the model on a separate test set.

Recommended metrics:

- **WER** - Word Error Rate
- **CER** - Character Error Rate
- **RTF** - Real-Time Factor

When reporting evaluation results, it is recommended to include the test dataset
name, number of audio files, total audio duration, device name, and
`transformers` version.

## Files

| File | Description |
| --- | --- |
| `config.json` | Whisper model configuration |
| `generation_config.json` | Generation and decoder settings |
| `preprocessor_config.json` | Audio preprocessing settings |
| `tokenizer_config.json` | Tokenizer configuration |
| `vocab.json`, `merges.txt` | Tokenizer vocabulary |
| `requirements.txt` | Minimal Python dependencies |
