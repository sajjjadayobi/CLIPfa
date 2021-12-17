from transformers import AutoModel, CLIPVisionModel

from ..config import TEXT_MODEL, IMAGE_MODEL
from .utils import clip_wraper_creator
from .data import item


vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_MODEL)
text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
assert text_encoder.config.hidden_size == vision_encoder.config.hidden_size

clip = clip_wraper_creator()
clip.text_model = text_encoder
clip.vision_model = vision_encoder


if __name__ == '__main__':
    out = clip(input_ids=item['input_ids'],
               attention_mask=item['attention_mask'],
               pixel_values=item['pixel_values'],
               return_loss=True)

    print('text and image embeddings: ',
          out.text_embeds.shape, out.image_embeds.shape)
    print('loss: ', out.loss)
    del out, item,
