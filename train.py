import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from network import ImageCaptioningModel

from tqdm import tqdm
from flickr_dataset import FlickrDataset, TextPadCollate


def train(model, train_dataloader, optimizer, criterion, n_epochs):
    for epoch_idx in range(n_epochs):
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
        
        print(f"Train Loss at epoch {epoch_idx}: {train_loss}")


if __name__ == "__main__":
    root_dir = "/l/vision/v5/sragas/DLS/ProjectData/Flickr8K/"
    transform_list = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FlickrDataset(root_dir+"Images/", root_dir+"captions.txt", transform_list)
    pad_index = dataset.vocabulary.str_to_idx["<PAD>"]

    train_dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True, pin_memory=True, collate_fn=TextPadCollate(pad_index))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocabulary)
    num_layers = 2
    learning_rate = 5e-4
    n_epochs = 100
    
    model = ImageCaptioningModel("resnet50", True, embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocabulary.str_to_idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_dataloader, optimizer, criterion, n_epochs)