from transformers import default_data_collator
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

from ..config import (tokenizer, BATCH_SIZE, IMAGE_SIZE, MEAN,
                      STD, MAX_LEN, DATA_FILE, TEST_SIZE)
from .utils import show_data


class CLIPDataset(Dataset):
    def __init__(self, image_paths: list, text: list, mode: str = 'train'):
        self.image_paths = image_paths
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=MAX_LEN, truncation=True)

        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])
        elif mode == 'test':
            self.augment = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {'input_ids': token.ids, 'attention_mask': token.attention_mask,
                'pixel_values': self.augment(Image.open(self.image_paths[idx]).convert('RGB'))}

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE)
    train_ds = CLIPDataset(image_paths=train_df.image.tolist(),
                           text=train_df.caption.tolist(), mode='train')
    test_ds = CLIPDataset(image_paths=test_df.image.tolist(),
                          text=test_df.caption.tolist(), mode='test')

    train_dl = DataLoader(train_ds, batch_size=2,
                          collate_fn=default_data_collator)
    for item in train_dl:
        print(item['input_ids'].shape)
        print(item['pixel_values'].shape)
        break

    show_data(item)
