import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image


class CLIPDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
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
        item = self.df.iloc[idx]
        image = self.augment(Image.open(item['image']))
        text = TOKENIZER(item['caption'], padding='max_length',
                         max_length=MAX_LEN, truncation=True, return_tensors='pt')
        
        return {'input_ids': text['input_ids'][0], 'attention_mask': text['attention_mask'][0], 'pixel_values': image}

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=0)
    train_ds = CLIPDataset(train_df, mode='train')
    test_ds = CLIPDataset(test_df, mode='test')

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
    for item in train_dl:
        print(item['input_ids'].shape)
        print(item['pixel_values'].shape)
        break
        
    show_data(item)
