# DeepSeek-Math

> Paper: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

DeepSeek-Math is a 7B parameter model that achieves 51.7% on the MATH benchmark, approaching frontier model performance through:

1. **Mathematical pre-training corpus** - 120B tokens curated from Common Crawl
2. **Group Relative Policy Optimization (GRPO)** - Efficient RL without a critic model
3. **Tool-integrated reasoning** - Program-of-Thought with Python execution

## Key Results

| Benchmark | DeepSeekMath-7B-RL | GPT-4 |
|-----------|-------------------|-------|
| MATH | 51.7% | 42.5% |
| GSM8K | 88.2% | 92.0% |

## Model Variants

- `deepseek-ai/deepseek-math-7b-base` - Continued pre-training on math corpus
- `deepseek-ai/deepseek-math-7b-instruct` - Supervised fine-tuning
- `deepseek-ai/deepseek-math-7b-rl` - GRPO reinforcement learning

## GRPO Algorithm

GRPO eliminates the expensive critic/value model by using group-relative baselines:

```python
# For each question, sample G outputs
outputs = [model.generate(question) for _ in range(G)]
rewards = [evaluate(o) for o in outputs]

# Normalize within group (no critic needed!)
advantages = (rewards - mean(rewards)) / std(rewards)

# Policy gradient update
loss = -sum(advantages * log_probs)
```

## Documentation

- [DEEP_DIVE.md](DEEP_DIVE.md) - Comprehensive technical analysis

## Citation

```bibtex
@article{deepseekmath2024,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Zhihong Shao and Peiyi Wang and Qihao Zhu and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```
