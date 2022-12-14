import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from network import ImageCaptioningModel

from tqdm import tqdm
from flickr_dataset import FlickrDataset, TextPadCollate

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

from collections import defaultdict
import wandb


# Logging
class Logger:
    def __init__(self, n_epochs) -> None:
        # self.log_dict = {"Training Loss": [], "Training Accuracy": []}
        self.log_dict = {"N_epochs": list(range(1, n_epochs+1))}
        self.n_epochs = n_epochs

    def log_metric(self, current_data: dict, epoch_idx: int) -> None:
        current_data = [(k, v) for k, v in current_data.items()][0]
        current_metric, current_value = current_data
        if current_metric not in self.log_dict:
            self.log_dict[current_metric] = [None] * self.n_epochs
            self.log_dict[current_metric][epoch_idx] = current_value
        else:
            self.log_dict[current_metric][epoch_idx] = current_value

    def return_non_empty_df(self):
        report = pd.DataFrame(self.log_dict)
        report = report.dropna()
        return report

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log_dict)

    def save_log(self, path: str) -> None:
        log_df = pd.DataFrame(self.log_dict)
        log_df.to_csv(path)


def seed_everything(seed_value=15):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # random.seed(seed_value)

def caption_image(model, image, vocabulary, max_length=50):
        result_caption_idx = []
        with torch.no_grad():
            # Send image features first
            feats = model.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hidden, states = model.decoder.lstm(feats, states)
                out = model.decoder.linear(hidden.squeeze(0))
                pred = out.argmax(1)

                result_caption_idx.append(pred.item())
                # After the 1st stage, the features of second stage becomes the caption from the 1st stage.
                feats = model.decoder.embed(pred).unsqueeze(0)
                if vocabulary.idx_to_str[pred.item()] == "<END>":
                    break
        return [vocabulary.idx_to_str[idx] for idx in result_caption_idx]



def training_loop(model, train_dataloader, valid_dataloader, optimizer, criterion, n_epochs, vocabulary, name):
    logger = Logger(n_epochs)

    n_valid_plot = 4
    rand_indices_for_plotting = [np.random.randint(0, len(valid_dataloader)) for i in range(n_valid_plot)]

    for epoch_idx in range(n_epochs):
        examples = defaultdict(str)

        model.train()
        train_loss = 0
        for image_batch, caption_batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            image_batch = image_batch.to(device)
            caption_batch = caption_batch.to(device)
            out = model(image_batch, caption_batch[:-1])
            # (seq_length, N, vocab_size) -> (seq_length, N)
            loss = criterion(out.reshape(-1, out.shape[2]), caption_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        model.eval()
        valid_bleu = 0
        for idx, (image, caption_embed_gt) in enumerate(tqdm(valid_dataloader)):
            image = image.to(device)
            caption_embed_gt = caption_embed_gt.to(device)
            out_caption = caption_image(model, image, vocabulary)
            
            pred_caption = out_caption[1:-1]
            gt_caption = [[vocabulary.idx_to_str[embed] for embed in caption_embed_gt.squeeze(0).tolist()][1:-1]]

            valid_bleu += sentence_bleu(gt_caption, pred_caption)

            if idx in rand_indices_for_plotting:
                pred_caption = " ".join([cap for cap in pred_caption])
                gt_caption = " ".join([cap for cap in gt_caption[0]])

                caption = f"GT: {gt_caption}\nPred: {pred_caption}"
                example = wandb.Image(image.squeeze(0), caption=caption)
                examples[f"Example {idx}"] = example

        valid_bleu /= len(valid_dataloader)

        logger.log_metric({f"Train Loss": train_loss}, epoch_idx)
        logger.log_metric({f"Valid BLEU": valid_bleu}, epoch_idx)

        log_dict_wandb = {
            "Train Loss": train_loss,
            "Valid BLEU": valid_bleu,
        }

        log_dict_wandb.update(dict(examples))
        wandb.log(log_dict_wandb)

        print(f"Train Loss at epoch {epoch_idx}: {train_loss}")
        print(f"Valid BLEU at epoch {epoch_idx}: {valid_bleu}")
        
        logs = logger.return_non_empty_df()
        logs.to_csv(f"./{name}_model_report.csv")

        torch.save(model.state_dict(), f"./{name}_model_img_cap.pth")



if __name__ == "__main__":
    root_dir = "/l/vision/v5/sragas/DLS/ProjectData/Flickr8K/"
    transform_list = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FlickrDataset(root_dir+"Images/", root_dir+"captions.txt", transform_list)
    pad_index = dataset.vocabulary.str_to_idx["<PAD>"]

    # Consider all captions for an image except for the last one in training.
    # In our case, there are 5 captions for each image.
    # We are considering the 1st 4 captions for an image.
    train_indices = [idx for idx in range(len(dataset)) if (idx + 1) % 5 != 0]
    valid_indices = [idx for idx in range(len(dataset)) if (idx + 1) % 5 == 0]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_extractor_name = "resnet50"
    # feat_extractor_name = "resnext50_32x4d"
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocabulary)
    num_layers = 2
    learning_rate = 5e-4
    n_epochs = 100
    batch_size = 64
    seed_value = 15
    pretrained = True
    name = f"{feat_extractor_name}_{pretrained}_embed_{embed_size}_hidden_{hidden_size}_lstm_{num_layers}_B_{batch_size}"
    seed_everything(seed_value)

    config = {
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "feature_extractor": feat_extractor_name,
        "pretrained": pretrained,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocabulary_size" : vocab_size,
        "seed": seed_value,
    }

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=8,
                                  shuffle=True, pin_memory=True, collate_fn=TextPadCollate(pad_index))

    valid_dataloader = DataLoader(valid_dataset, batch_size=1,
                                  shuffle=False, pin_memory=True)
                                  # collate_fn=TextPadCollate(pad_index))

    wandb.init(project="Image-Captioning", entity="asrimanth", config=config)

    model = ImageCaptioningModel(feat_extractor_name, pretrained,
                                 embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocabulary.str_to_idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_loop(model, train_dataloader, valid_dataloader, optimizer, criterion, n_epochs, dataset.vocabulary, name)
