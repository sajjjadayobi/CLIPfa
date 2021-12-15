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
- create demo

## Demo
- Colab notebooks
- Huggingface 🤗 spaces
- Torchserv 🥘

```python
from transformers import CLIPVisionModel, RobertaModel
from clipfa.utils import get_image_embeddings, get_text_embedding, most_similar

vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')

image_embeddings = get_image_embeddings(vision_encoder, test_dl, device='cuda')
text_embedding = get_text_embedding(text_encoder, query='موز', device='cuda')
```

## Datasets: 400K
- Flicker30K (25K)
- MS-COCO (50K!)
- WiT (125K!)
- Divar (200K!)
- [CC](https://ai.google.com/research/ConceptualCaptions/download)
- [image2ds](https://github.com/rom1504/img2dataset)

## Applications
- image search using description
- zero shot image classification
- Anology
