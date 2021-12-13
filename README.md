# CLIPfa: Connecting Farsi Text and Images

## Demo
- Colab notebooks
- Huggingface 🤗 spaces
- Torchserv 🥘

```python
from transformers import CLIPVisionModel, RobertaModel

vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
```

## Applications
- image search using image
- image search using description
- zero shot image classification
