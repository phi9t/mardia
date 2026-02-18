# Qwen3-TTS (Educational)

This directory implements a paper-grounded, educational reconstruction of the key ideas in Qwen3-TTS:

1. 12Hz discrete speech tokenization interface.
2. Dual-track LM generation structure.
3. Lightweight causal ConvNet decode path.
4. Single-speaker SFT recipe and data format.

Use this as a code-first companion to the technical report, not as a production replacement for official models.

## Quick start

```bash
python -m qwen3_tts.validation.run_educational_smoke
python -m qwen3_tts.validation.compare_structure
```

## Files

- `DEEP_DIVE.md`: narrative walkthrough.
- `PAPER_MAP.md`: paper section to code map.
- `architecture/`: model components.
- `training/`: prepare + train recipe.
- `data/`: Hugging Face conversion helpers.
- `validation/`: structural and behavior checks.
