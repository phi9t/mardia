# Dataset Format

Each sample follows:

```json
{
  "image": ["img0.jpg"],
  "video": [],
  "conversations": [
    {"from":"human","value":"<image> Describe this chart."},
    {"from":"assistant","value":"..."}
  ]
}
```

Placeholder rules:

1. Number of `<image>` tags must match supplied images.
2. Number of `<video>` tags must match supplied videos.
