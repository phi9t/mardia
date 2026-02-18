# Training Recipe (Educational)

1. Prepare JSONL in `audio`, `text`, `ref_audio` format.
2. Run `prepare_data.py` to add `audio_codes`.
3. Run `train_sft_single_speaker.py` for a compact SFT loop.

This tracks the public single-speaker fine-tuning style in the official repo.
