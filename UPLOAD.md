# Upload Instructions

This folder is prepared for upload to Hugging Face as a model repository.

1. Install the CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

2. Login:

```bash
huggingface-cli login
```

3. Create a model repo on Hugging Face, for example:

```bash
huggingface-cli repo create uz-whisper-small-stt --type model
```

4. Upload from this folder:

```bash
cd C:\Users\hp\Desktop\stt_model\hf_ready_model
huggingface-cli upload YOUR_USERNAME/uz-whisper-small-stt . --repo-type model
```

Change `YOUR_USERNAME/uz-whisper-small-stt` to your own Hugging Face username
and repository name.
