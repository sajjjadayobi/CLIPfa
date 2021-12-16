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

## How to use?
You can use these models of the shelf. Both models create vectors with 768 dimention.
```python
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer
vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
```

## Demo:
The followings are just some use cases of CLIP model.
- use `pip install -q git+https://github.com/sajjjadayobi/clipfa.git`
```python
from clipfa import CLIPDemo
demo = CLIPDemo(vision_encoder, text_encoder, tokenizer)
```
### Image Search:
```python
image_paths = []
demo.image_search(query='اسب', image_paths=image_paths)
```
### Zero Shot Image Classification:
```python
class_list = ['چند مرد','موز' ,'بیل']
demo.zero_shot(image_path='workers.jpg', class_list=class_list)
```
### Analogy: 
```python
image_paths = []
demo.anology('sunset.jpg', image_paths=image_paths, additional_text='دریا')
```

- Colab notebooks
- Huggingface 🤗 spaces


## Datasets: 300K
- Flicker30K (25K)
- MS-COCO (50K!)
- WiT (125K!)
- Divar (100K!)
- [image2ds](https://github.com/rom1504/img2dataset)
