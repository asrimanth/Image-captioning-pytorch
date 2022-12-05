import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import spacy

import pandas as pd
from collections import defaultdict

SPACY_ENG = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freq_thresh):
        self.idx_to_str = {0:"<PAD>", 1:"<START>", 2:"<END>", 3:"<UNK>"}
        self.str_to_idx = {val:key for key, val in self.idx_to_str.items()}
        self.freq_thresh = freq_thresh
    
    def __len__(self):
        return len(self.idx_to_str)

    @staticmethod
    def tokenizer_english(text):
        return [token.text.lower() for token in SPACY_ENG.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = defaultdict(int)
        # Indices 0-3 are taken in self.idx_to_str
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_english(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_thresh:
                    self.str_to_idx[word] = idx
                    self.idx_to_str[idx] = word
                    idx += 1

    def numericalize(self, text):
        # Take text and convert it into numerical values
        tokenized_text = self.tokenizer_english(text)
        return [
            self.str_to_idx[token] if token in self.str_to_idx else self.str_to_idx["<UNK>"] 
            for token in tokenized_text
        ]


# -------------------- DOWNLOAD LINK: https://github.com/goodwillyoga/Flickr8k_dataset --------------------
class FlickrDataset(Dataset):
    def __init__(self, image_dir_path, caption_file_path, transform=None, freq_thresh=5):
        self.image_dir_path = image_dir_path
        self.caption_file_path = caption_file_path
        self.transform = transform
        self.image_caption_mapping = pd.read_csv(caption_file_path, names=["ImageName", "Caption"], header=0)
        self.image_caption_mapping.rename(columns={"image": "ImageName", "caption": "Caption"})
        self.image_caption_mapping["ImageName"] = self.image_caption_mapping["ImageName"].str.split("#", 1, expand=True)[0]
        self.image_caption_mapping["ImagePath"] = self.image_dir_path + self.image_caption_mapping["ImageName"].astype(str)
        self.image_caption_mapping.to_csv("temp.csv")

        self.image_list = list(self.image_caption_mapping["ImageName"])
        self.image_path_list = list(self.image_caption_mapping["ImagePath"])
        self.caption_list = list(self.image_caption_mapping["Caption"])

        self.vocabulary = Vocabulary(freq_thresh)
        self.vocabulary.build_vocabulary(self.caption_list)

    def __len__(self):
        return len(self.image_caption_mapping)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        caption = self.caption_list[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Convert each word to an index which is in our vocabulary.
        numerical_caption = [self.vocabulary.str_to_idx["<START>"]]
        numerical_caption += self.vocabulary.numericalize(caption)
        numerical_caption += [self.vocabulary.str_to_idx["<END>"]]

        return image, torch.LongTensor(numerical_caption)


class TextPadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        # print(f"Target shape before padding: {[item[1].shape for item in batch]}")
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return images, targets


if __name__ == "__main__":
    root_dir = "/l/vision/v5/sragas/DLS/ProjectData/Flickr8K/"
    transform_list = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset = FlickrDataset(root_dir+"Images/", root_dir+"captions.txt", transform_list)
    pad_index = dataset.vocabulary.str_to_idx["<PAD>"]

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True, pin_memory=True, collate_fn=TextPadCollate(pad_index))

    for image, encoded_text in dataloader:
        print(image.shape)
        print(encoded_text.shape)
