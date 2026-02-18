# Qwen3-TTS Deep Dive (Code-First)

## What this captures

This implementation captures the essence of the Qwen3-TTS report and repo runtime API:

1. Discrete codec-first generation flow.
2. Dual-track generation idea for low-latency and offline modes.
3. Causal ConvNet waveform reconstruction path.
4. Task-level control modes: custom voice, voice design, voice clone.

## Reading order

1. `architecture/model.py`
2. `architecture/dual_track_lm.py`
3. `architecture/speech_decoder_convnet.py`
4. `architecture/tokenizer_12hz.py`
5. `training/train_sft_single_speaker.py`

## Notes

- This is intentionally compact and inspectable.
- It does not attempt checkpoint-level parity with official release artifacts.
- Validation is structural and behavior-oriented.
