<span align="center">
        <a href="https://huggingface.co/spaces/SajjadAyoubi/CLIPfa-Demo"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=HF Demo&color=blue"></a>
    <a href="https://huggingface.co/SajjadAyoubi/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Models&color=red"></a>
</span>

# CLIPfa: Connecting Farsi Text and Images
OpenAI released [`the paper Learning Transferable Visual Models From Natural Language Supervision`](https://arxiv.org/abs/2103.00020) in which they present the CLIP (Contrastive Language–Image Pre-training) model. This model is trained to connect text and images, by matching their corresponding vector representations using a contrastive learning objective. CLIP consists of two separate models, a vision encoder and a text encoder. These were trained on a wooping 400 Million images and corresponding captions. We have trained a Farsi (Persian) version of OpenAI's CLIP on a dataset of 400,000 (image, text) pairs. We used [`Farahani's RoBERTa-fa`](https://huggingface.co/m3hrdadfi/roberta-zwnj-wnli-mean-tokens) as the text encoder and [‍‍`ViT‍`](https://huggingface.co/openai/clip-vit-base-patch32) as the vision encoder from Original CLIP and finetuned them.

![CLIPfa image](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/clipfa.png)

It should be noted that only 400K pairs were used for this training, whereas 4 million pairs were used for the Original CLIP. Also, the training took 30 days across 592 GPUs powered by the V100 chip.
 

## How to use?
Both models generate vectors with 768 dimensions.
```python
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor
# download pre-trained models
vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')
text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
# define input image and input text
text = 'something'
image = PIL.Image.open('my_favorite_image.jpg')
# compute embeddings
text_embedding = text_encoder(**tokenizer(text, return_tensors='pt')).pooler_output
image_embedding = vision_encoder(**preprocessor(image, return_tensors='pt')).pooler_output
text_embedding.shape == image_embedding.shape
```

## Demo:
The followings are just some use cases of CLIPfa on 25K [`Unsplash images`](https://github.com/unsplash/datasets)
- use `pip install -q git+https://github.com/sajjjadayobi/clipfa.git`
```python
from clipfa import CLIPDemo
demo = CLIPDemo(vision_encoder, text_encoder, tokenizer)
demo.compute_text_embeddings(['گاو' ,'اسب' ,'ماهی'])
demo.compute_image_embeddings(test_df.image_path.to_list())
```
### Image Search:
```python
demo.image_search(query='غروب خورشید')
```
![](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/image_search.png)

```python
demo.image_search(query='جنگل در زمستان برفی')
```
![](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/forest%20in%20winter.png)

### Analogy: 
```python
demo.anology('sunset.jpg', additional_text='دریا')
```
![](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/analogy-sea.png)

```python
demo.anology('sunset.jpg', additional_text='برف')
```
![](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/analogy-snow.png)

### Zero Shot Image Classification:
```python
demo.zero_shot(image_path='apples.jpg')
```
- Provided labels with their probability for each image.

|                   گاو:36 , ماهی:22, اسب:42                   |                   گاو:41 , ماهی:23, اسب:36                   |                 گاو:26 , ماهی:**45**, اسب:27                 |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/horse.jpg) | ![image](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/cow.jpg) | ![image](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/fish.jpg) |

## Online Demo: [CLIPfa at Huggingface🤗 spaces](https://huggingface.co/spaces/SajjadAyoubi/CLIPfa-Demo)
We used a small set of images (25K) to keep this app almost real-time, but it's obvious that the quality of image search depends heavily on the size of the image database. 

![](https://github.com/sajjjadayobi/CLIPfa/blob/main/assets/hf-spaces.png)


## Dataset:
### Training v1 (400K)
We started with this question that how much the original Clip model depends on its big training dataset containing a lot of conceptual samples. Our model shows that It is possible to meet an acceptable enough target with only a little amount of data even though, It may not have known enough concepts and subjects to be used widely. Our model trained on a dataset gathered from different resources such as The Flickr30k, MS-COCO 2017, Google CCm3, ... . We used these datasets and translated them into the Persian language with a [`tool`](https://github.com/sajjjadayobi/CLIPfa/blob/main/clipfa/data/translation.py) prepared by ourselves. Using the Google Translate and Multilingual Similarity Check method we provided an automatic translator that has been given a list of English captions and filtered by the best translations.

### Available v2 (20M)
We started with the Persian(Farsi) subset of the [`LAION-2B-multi`](https://huggingface.co/datasets/laion/laion2B-multi) dataset and released a cleaned version called [`LAION-20M-fa`](https://huggingface.co/datasets/amir7d0/laion20M-fa). We used [`image2dataset`](https://github.com/rom1504/img2dataset)  to download LAION-20M-fa dataset images. Then we calculated the similarity of the image and text embeddings with CLIP-fa. After the similarity score was established we removed the pairs under the threshold we decided to use, i.e 0.50 for the Persian dataset. Then we used a small portion of LAION-20M-fa to fine-tune CLIP-fa v2 because of computation restrictions.
- It'd be appreciated if someone with access to some GPUs could train the model on the whole dataset and share it with the community.


## Training: <a href="https://colab.research.google.com/github/sajjjadayobi/CLIPfa/blob/main/notebook/CLIPfa_Training.ipynb"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=CLIPfa Training&color=white"></a>
Any dataset can be used with little change by the [`training code`](https://github.com/sajjjadayobi/CLIPfa/tree/main/clipfa). CLIPfa can be trained with other encoders as long as they have the same hidden size at the last layer.  In [`this`](https://github.com/sajjjadayobi/CLIPfa/blob/main/notebook/CLIPfa_Training.ipynb) notebook I used [`training code`](https://github.com/sajjjadayobi/CLIPfa/tree/main/clipfa) to train a small CLIP on translated [‍`flickr30K`](https://www.kaggle.com/sajjadayobi360/flickrfa) dataset.


## Citation: ↩️
If you have a technical question regarding the model, code or publication, create an issue in the repository.
we didn't publish any papers on the work. However, if you did, please cite us properly with an entry like one below.
```bibtex
@misc{CLIPfa,
  author          = {Sajjad Ayoubi, Navid Kanaani, Amir Ahmadi},
  title           = {CLIPfa: Connecting Farsi Text and Images},
  year            = 2022,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/CLIPfa}},
}
```
> Made with ❤️ in my basement🤫
