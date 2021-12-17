<span align="center">
    <a href="https://huggingface.co/SajjadAyoubi/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=SajjadAyoubi&color=yellow"></a>
    <a href="https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Demo&logo=Google%20Colab&color=f9ab00"></a>
</span>

# CLIPfa: Connecting Farsi Text and Images
CLIP (Contrastive Language-Image Pre-Training) is the first multimodal (in this case, vision and text) model tackling computer vision and was recently released by OpenAI on January 5, 2021. We've trained a Tiny Farsi(Persian) version of [OpenAI's CLIP](https://openai.com/blog/clip/) on a crawled dataset with 300,000 (image, text) pairs. We used RoBerta-fa and Original CLIP's ViT as our starting point.Both models create vectors with 768d and same as paper we used contrastive loss. 
![](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/clipfa.png)
- Keep it in mind that, this model was trained for 5 epochs only on 300K pairs whereas the Original CLIP was traind on 4m pairs and The training process took 30 days across 592 V100 GPUs.


OpenAI recently released the paper Learning Transferable Visual Models From Natural Language Supervision in which they present the CLIP (Contrastive Language–Image Pre-training) model. This model is trained to connect text and images, by matching their corresponding vector representations using a contrastive learning objective. CLIP consists of two separate models, a visual encoder and a text encoder. These were trained on a wooping 400 Million images and corresponding captions.

## How to use?
You can use these models of the shelf. Both models create vectors with 768 dimention.
```python
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer
vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
vision_preprocessor = 
text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
```

## Demo: Huggingface 🤗 spaces
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

## Datasets: 300K
- Flicker30K (25K)
- MS-COCO (50K!)
- WiT (125K!)
- VizWiz (20!)
- Divar (100K!)
- [image2ds](https://github.com/rom1504/img2dataset)


## Contact us: :open_hands:
If you have a technical question regarding the model, code or publication, create an issue in the repository.

## Citation: ↩️
we didn't publish any papers on the work. However, if you did, please cite us properly with an entry like one below.
```bibtex
@misc{ParsBigBird,
  author          = {Ayoubi, Sajjad},
  title           = {CLIPfa: Connecting Farsi Text and Images},
  year            = 2022,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/CLIPfa}},
}
```


## Progress
- training code ✅
- hypersearch ✅
- Flicker30K ✅
- Translate COCO
- Move Code on Server
- Downloada WiT images
- Crwal divar
- Train model
