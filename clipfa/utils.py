import torch
import matplotlib.pyplot as plt
import gc
import multiprocessing
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


def get_image_embeddings(clip, datalodear, device='cuda'):
    clip.eval().to(device)
    embeddings = []
    with torch.no_grad():
        for item in tqdm(datalodear):
            image_embedding = clip.get_image_features(
                pixel_values=item['pixel_values'].to(device))
            embeddings.append(image_embedding)
    return torch.cat(embeddings)


def get_text_embedding(clip, query='football', device='cuda'):
    clip.eval().to(device)
    tokens = TOKENIZER(query, return_tensors='pt')
    with torch.no_grad():
        text_embedding = clip.get_text_features(input_ids=tokens["input_ids"].to(device),
                                                attention_mask=tokens["attention_mask"].to(device))
    return text_embedding


def most_similar_images(image_embeddings, text_embeddings, images):
    values, indices = torch.cosine_similarity(
        image_embeddings, text_embeddings).sort(descending=True)
    return images[indices.cpu()]
