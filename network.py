import torch
from torch import nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, feature_extractor_name, embed_size, pretrained=True):
        super(EncoderCNN, self).__init__()
        self.pretrained = pretrained
        if pretrained:
            weights="IMAGENET1K_V2"
        else:
            weights = None

        # self.encoder = torch.hub.load("pytorch/vision", feature_extractor_name, weights=weights)
        self.encoder = models.get_model(feature_extractor_name, weights=weights)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, images):
        encoded_feat = self.encoder(images)
        for name, param in self.encoder.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.pretrained

        return self.dropout(self.relu(encoded_feat))


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, _ = self.lstm(embeddings)
        out = self.linear(hidden)
        return out


class ImageCaptioningModel(nn.Module):
    def __init__(self, feature_extractor_name, pretrained, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(feature_extractor_name, embed_size, pretrained)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        out = self.decoder(features, captions)
        return out

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption_idx = []

        with torch.no_grad():
            # Send image features first
            feats = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hidden, states = self.decoder.lstm(feats, states)
                out = self.decoder.linear(hidden.squeeze(0))
                pred = out.argmax(1)

                result_caption_idx.append(pred.item())
                # After the 1st stage, the features of second stage becomes the caption from the 1st stage.
                feats = self.decoder.embed(pred).unsqueeze(0)
                
                if vocabulary.idx_to_str[pred.item()] == "<END>":
                    break
        return [vocabulary.idx_to_str[idx] for idx in result_caption_idx]
