# Qwen3-VL Paper Map

## Section -> Code

1. Interleaved-MRoPE
   - `architecture/interleaved_mrope.py`
   - PaperRef: stronger spatial-temporal position modeling across image/video.

2. DeepStack visual fusion
   - `architecture/deepstack_adapter.py`
   - PaperRef: multi-level ViT feature integration and alignment.

3. Text-timestamp alignment
   - `architecture/timestamp_alignment.py`
   - PaperRef: text-conditioned temporal grounding for video.

4. Multimodal dense architecture path
   - `architecture/vl_model.py`
   - PaperRef: dense model path suitable for smaller variants.

5. Training data formatting and multimodal placeholders
   - `training/prepare_data.py`
   - `training/collator.py`

## Explicit omissions

1. No MoE expert routing implementation.
2. No full production vision processor or all decoder backends.
3. No exact benchmark reproduction.
