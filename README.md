# CLIPfa: Connecting Farsi Text and Images


## Progress
- training code ✅
- hypersearch ✅
- move on Server
- Downloada WiT images
- crwal divar
- train model
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

## Applications
- image search using image
- image search using description
- zero shot image classification
