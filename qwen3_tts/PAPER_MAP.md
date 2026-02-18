# Qwen3-TTS Paper Map

## Section -> Code

1. Tokenizer / low-bitrate discrete speech representation
   - `architecture/tokenizer_12hz.py`
   - PaperRef: 12Hz multi-codebook tokenizer path.

2. Dual-track generation architecture
   - `architecture/dual_track_lm.py`
   - `architecture/model.py`
   - PaperRef: dual-track LM for realtime/non-realtime generation behavior.

3. Lightweight causal decoder for low latency
   - `architecture/speech_decoder_convnet.py`
   - PaperRef: lightweight causal ConvNet waveform reconstruction path.

4. Controllable generation tasks (custom/design/clone)
   - `architecture/generation.py`
   - PaperRef: instruction-conditioned control and voice-clone modes.

5. Fine-tuning recipe (single-speaker public path)
   - `training/prepare_data.py`
   - `training/train_sft_single_speaker.py`
   - PaperRef: open fine-tuning path centered on base model and prepared audio codes.

## Explicit omissions

1. No attempt at numerical reproduction of official checkpoints.
2. No block-wise streaming server implementation.
3. No production tokenizer internals from closed/released binaries.
