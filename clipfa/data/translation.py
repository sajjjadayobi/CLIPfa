# necessary packages for translation
# !pip install -q sentence_transformers
# !pip install -q mtranslate

from sentence_transformers import SentenceTransformer
from mtranslate import translate
from tqdm import tqdm
import numpy as np
import torch


class MultiLangSimilarity():
    def __init__(self, device:str='cpu'):
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)

    # this isn't the fastest way to do this (it can be done in batches)
    def __call__(self, a, b):
        a, b = self.model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return torch.dot(a, b).item()

class Translator():
    def __init__(self, min_score:float=.9, device:str='cpu'):
        self.min_score = min_score
        self.similar = MultiLangSimilarity(device=device)

    def __call__(self, sentences:list):
        outputs = []
        for sentence in tqdm(sentences):
            target = translate(sentence, from_language='en', to_language='fa')
            score = self.similar(sentence, target)
            # ignore translation with score less than min_score
            if score >= self.min_score:
                outputs.append(target)
            else:
                outputs.append('[NAN]')
        return outputs


if __name__ == '__main__':
    # Split data into some chunks (5k is recommended.  bigger than this you may get HTTP error)
    caption_chunks = np.load('caption_5k_chunks.npy')
    translator = Translator(min_score=0.85, device='cpu')
    translations = []
    # loop through chunks
    for chunk in range(len(caption_chunks)): 
      # Feed translator with english captions
      translations.append(translator(sentences=caption_chunks[chunk])) 
