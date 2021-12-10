
from torch import nn
from transformers import AutoModel, CLIPModel

from config import TEXT_MODEL, IMAGE_MODEL
from data import item


clip = CLIPModel.from_pretrained(IMAGE_MODEL)
clip.text_model = AutoModel.from_pretrained(TEXT_MODEL).base_model
# convert text_projection to be compatible with new text encoder
clip.text_projection = nn.Linear(clip.text_model.config.hidden_size,
                                 clip.projection_dim, bias=False)


if __name__ == '__main__':
    out = clip(input_ids=item['input_ids'],
               attention_mask=item['attention_mask'],
               pixel_values=item['pixel_values'],
               return_loss=True)

    print('text and image embeddings: ',
          out.text_embeds.shape, out.image_embeds.shape)
    print('loss: ', out.loss)
