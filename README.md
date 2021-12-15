# CLIPfa: Connecting Farsi Text and Images


## Progress
- training code ✅
- hypersearch ✅
- Flicker30K ✅
- Translate COCO
- Move Code on Server
- Downloada WiT images
- Crwal divar
- Train model
- convert to package

## Demo
- Colab notebooks
- Huggingface 🤗 spaces

```python
from transformers import CLIPVisionModel, RobertaModel
from clipfa.utils import get_image_embeddings, get_text_embedding, most_similar

vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
```

## Datasets: 300K
- Flicker30K (25K)
- MS-COCO (50K!)
- WiT (125K!)
- Divar (100K!)
- [image2ds](https://github.com/rom1504/img2dataset)
