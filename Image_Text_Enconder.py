import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import math
from dataloader_image_text import build_vocab, JiebaCustomDataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=True):
        super(ImageEncoder, self).__init__()
        # 使用 ResNet50 提取图像特征
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 去掉最后的分类层
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 输出 [B, 2048, 1, 1]
        self.fc = nn.Linear(2048, embed_dim)  # 投影到 embed_dim

    def forward(self, images):
        features = self.model(images)           # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        features = self.fc(features)            # [B, embed_dim]
        features = nn.functional.normalize(features, dim=1)  # L2 归一化
        return features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class MyTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, max_len=32, num_layers=2, num_heads=8, hidden_dim=2048, dropout=0.1):
        super(MyTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)  # 投影到最终特征
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] (1 表示有效，0 表示 PAD)
        """
        x = self.embedding(input_ids)  # [B, L, D]
        x = self.pos_encoder(x)        # 加位置编码

        if attention_mask is not None:
            # 1 表示有效 token，0 表示 padding
            src_key_padding_mask = (attention_mask == 0)  # True 表示 padding
        else:
            src_key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, D]

        cls_feature = x[:, 0, :]  # CLS token
        cls_feature = F.normalize(self.fc(cls_feature), dim=1)
        return cls_feature


if __name__ == "__main__":
    # 假设你的词表长度是 10000
    vocab_size = 81
    batch_size = 4
    seq_len = 16
    csv_file = "./dataset/test_ImageWordData.csv"
    img_dir = "./dataset/test_ImageData"
    word2id, id2word = build_vocab()
    dataset = JiebaCustomDataset(csv_file, img_dir, word2id, max_length=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    # 模拟输入
    for batch in dataloader:
        images, input_ids, attention_mask = batch
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_encoder = MyTextEncoder(vocab_size=vocab_size)
        text_features = text_encoder(input_ids, attention_mask)
        image_encoder = ImageEncoder()
        image_features = image_encoder(images)
