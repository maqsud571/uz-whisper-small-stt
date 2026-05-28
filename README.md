
# Uzbek Whisper Small STT

`maqsudxo1ja/uz-whisper-small-stt` - o'zbek tilidagi nutqni matnga aylantirish
uchun tayyorlangan Whisper Small asosidagi Speech-to-Text modeli.

Model Hugging Face `transformers` kutubxonasi bilan ishlaydi va Uzbek (`uz`)
transkripsiyasi uchun default sozlangan. Asosiy model:
[`openai/whisper-small`](https://huggingface.co/openai/whisper-small).

## Artefaktlar

- **Model:** https://huggingface.co/maqsudxo1ja/uz-whisper-small-stt
- **Vazifa:** Automatic Speech Recognition (ASR), ya'ni nutqdan matnga o'tkazish
- **Default til:** Uzbek (`uz`)
- **Default task:** `transcribe`
- **Model formati:** `safetensors`
- **Kutubxona:** `transformers`
- **Litsenziya:** Apache-2.0

## Model haqida

Ushbu model Whisper arxitekturasining `small` variantiga asoslangan
encoder-decoder modelidir. Audio kirish signali avval 16 kHz formatga
moslanadi, keyin log-Mel spektrogramma ko'rinishiga o'tkaziladi. Encoder audio
belgilarini kodlaydi, decoder esa ularni matn tokenlariga aylantiradi.

Model konfiguratsiyasida o'zbek tili va transkripsiya rejimi oldindan
belgilangan:

- `language`: `uz`
- `task`: `transcribe`
- `return_timestamps`: `false`
- `sampling_rate`: `16000`
- `chunk_length`: `30`
- `n_samples`: `480000`

## Sekundlar bo'yicha ishlashi

Whisper modeli odatda audioni 30 soniyalik kontekst oynasida qayta ishlaydi.
Ushbu repodagi `preprocessor_config.json` qiymatlariga ko'ra:

| Parametr | Qiymat | Ma'nosi |
| --- | ---: | --- |
| `sampling_rate` | 16000 Hz | Audio 1 soniyada 16000 sample ko'rinishida beriladi |
| `chunk_length` | 30 soniya | Model uchun asosiy audio oynasi |
| `n_samples` | 480000 | 30 soniya x 16000 sample |
| `feature_size` | 80 | Log-Mel feature soni |
| `hop_length` | 160 | Taxminan har 10 ms uchun feature qadam |

Qisqa audioda, masalan 5-10 soniyalik faylda, model shu audio bo'lagini qayta
ishlaydi va bitta matn natijasini qaytaradi. 30 soniyadan uzun audioda esa
audio bo'laklarga ajratib qayta ishlanishi kerak. Amaliy ishlatishda
`chunk_length_s=30` va chegaralardagi so'zlar yo'qolmasligi uchun
`stride_length_s=5` ishlatish tavsiya etiladi.

Inference vaqti audioning uzunligiga, CPU/GPU turiga, RAM/VRAM holatiga va
batch sozlamalariga bog'liq. Tezlikni baholash uchun **Real-Time Factor (RTF)**
ishlatiladi:

```text
RTF = inference_seconds / audio_seconds
```

- `RTF < 1.0` - model real vaqtdan tezroq ishlayapti.
- `RTF = 1.0` - 30 soniya audio taxminan 30 soniyada qayta ishlanadi.
- `RTF > 1.0` - qayta ishlash audio uzunligidan sekinroq.

## O'rnatish

```bash
pip install -r requirements.txt
```

Yoki alohida o'rnatish:

```bash
pip install "transformers>=4.45.2" torch torchaudio safetensors
```

## Oddiy ishlatish

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

## To'g'ridan-to'g'ri `WhisperProcessor` bilan ishlatish

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

## Kutiladigan audio formati

- Tavsiya etilgan sample rate: 16 kHz
- Kanal: mono afzal
- Formatlar: `wav`, `mp3`, `flac` va `torchaudio`/backend qo'llaydigan boshqa
  audio formatlar
- Shovqinsiz, yaqin mikrofon orqali yozilgan audio odatda yaxshiroq natija
  beradi

## Baholash

Bu model kartasida hali tasdiqlangan WER/CER natijalari keltirilmagan.
Professional benchmark uchun modelni alohida test to'plamida baholang.

Tavsiya etiladigan metrikalar:

- **WER** - Word Error Rate
- **CER** - Character Error Rate
- **RTF** - Real-Time Factor

Baholashda test dataset nomi, audio soni, umumiy audio davomiyligi, qurilma
nomi va `transformers` versiyasini ko'rsatish tavsiya etiladi.

## Fayllar

| Fayl | Tavsif |
| --- | --- |
| `config.json` | Whisper model konfiguratsiyasi |
| `generation_config.json` | Generatsiya va decoder sozlamalari |
| `preprocessor_config.json` | Audio preprocessing sozlamalari |
| `tokenizer_config.json` | Tokenizer konfiguratsiyasi |
| `vocab.json`, `merges.txt` | Tokenizer lug'ati |
| `requirements.txt` | Minimal Python kutubxonalar |
