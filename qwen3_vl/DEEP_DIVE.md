# Qwen3-VL Deep Dive (Code-First)

## What this captures

1. Interleaved text-image-video position indexing via an MRoPE-like mechanism.
2. Multi-level visual feature fusion inspired by DeepStack.
3. Timestamp-aware video-text alignment scoring path.
4. Practical multimodal data formats with `<image>` and `<video>` placeholders.

## Reading order

1. `architecture/vl_model.py`
2. `architecture/interleaved_mrope.py`
3. `architecture/deepstack_adapter.py`
4. `architecture/timestamp_alignment.py`
5. `training/prepare_data.py`

## Notes

- Educational implementation, not production parity.
- Correctness checks are structural/behavioral.
