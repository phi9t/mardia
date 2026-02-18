# Dataset Format

Each line in input JSONL:

```json
{"audio":"path/to/utt.wav","text":"utterance","ref_audio":"path/to/ref.wav"}
```

Prepared JSONL adds:

```json
{"audio":"...","text":"...","ref_audio":"...","audio_codes":[[1,2,...],[...]]}
```
