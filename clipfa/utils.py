import torch
import matplotlib.pyplot as plt
from torch import nn
import gc
import multiprocessing
from transformers import CLIPConfig, CLIPModel
from tqdm import tqdm

from config import MEAN, STD, TOKENIZER


def clear_gpu():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


def optimal_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value


def show_data(batch, idx=0):
    # show image
    img = batch['pixel_values'][idx].permute(1, 2, 0)
    img = STD * img + MEAN
    print('Image shape: ', img.shape)
    plt.imshow(img)
    # show text
    text = TOKENIZER.decode(batch['input_ids'][idx],  skip_special_tokens=True)
    print('Text: ', text)


def get_image_embeddings(image_encoder, datalodear, device='cuda'):
    image_encoder.eval().to(device)
    embeddings = []
    with torch.no_grad():
        for item in tqdm(datalodear):
            image_embedding = image_encoder.forward(
                pixel_values=item['pixel_values'].to(device)).pooler_output
            embeddings.append(image_embedding)
    return torch.cat(embeddings)


def get_text_embedding(text_encoder, query='football', device='cuda'):
    text_encoder.eval().to(device)
    tokens = TOKENIZER(query, return_tensors='pt')
    with torch.no_grad():
        text_embedding = text_encoder(input_ids=tokens["input_ids"].to(device),
                                      attention_mask=tokens["attention_mask"].to(device)).pooler_output
    return text_embedding


def most_similar_images(image_embeddings, text_embeddings, images):
    values, indices = torch.cosine_similarity(
        image_embeddings, text_embeddings).sort(descending=True)
    return images[indices.cpu()]


def clip_wraper_creator():
    """create a dummy CLIPModel to wrap text and vision encoders in order to use CLIPTrainer"""
    config = {'num_hidden_layers': 0,
              'max_position_embeddings': 0,
              'vocab_size': 0,
              'hidden_size': 1,
              'patch_size': 1,
              }
    DUMMY_CONFIG = CLIPConfig(text_config_dict=config,
                              vision_config_dict=config)
    clip = CLIPModel(config=DUMMY_CONFIG)
    # convert projectors to Identity
    clip.text_projection = nn.Identity()
    clip.visual_projection = nn.Identity()
    return clip
