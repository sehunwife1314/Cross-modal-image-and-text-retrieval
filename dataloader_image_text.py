import pandas as pd
import jieba
from collections import Counter
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import re
from const import  punctuation


def build_vocab(csv_file="./dataset/ImageWordData.csv"):
    # 读取 CSV
    df = pd.read_csv(csv_file)

    # 编译正则，用来过滤标点和空白字符
    punct_pattern = re.compile(f"[{re.escape(punctuation)}]")

    counter = Counter()

    for caption in df['caption']:
        # 分词
        words = jieba.lcut(str(caption))
        # 去掉标点和空白字符
        words = [w for w in words if w.strip() and not punct_pattern.fullmatch(w)]
        counter.update(words)

    # 建立词表，前 3 个为特殊 token
    word2id = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
    for i, word in enumerate(counter.keys(), start=3):
        word2id[word] = i

    id2word = {v: k for k, v in word2id.items()}

    print("词表总大小:", len(word2id))  # 总共多少个词
    return word2id, id2word


class JiebaCustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, word2id, max_length=32, img_transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.word2id = word2id
        self.max_length = max_length
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_id']}"

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        words = [w for w in jieba.cut(row['caption']) if w.strip() and w not in punctuation]
        if len(words) == 0:
            words = ["[PAD]"]  # 保证至少有一个 token

        input_ids = [self.word2id.get(w, self.word2id["[PAD]"]) for w in words]
        attention_mask = [1] * len(input_ids)

        # 添加特殊 token
        input_ids = [self.word2id["[CLS]"]] + input_ids[:self.max_length - 2] + [self.word2id["[SEP]"]]
        attention_mask = [1] + attention_mask[:self.max_length - 2] + [1]

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.word2id["[PAD]"]] * pad_len
            attention_mask += [0] * pad_len

        return {
            "image": img,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


if __name__ == "__main__":
     csv_file = "./dataset/ImageWordData.csv"
     img_dir = "./dataset/ImageData"
     word2id, id2word = build_vocab(csv_file=csv_file)
     dataset = JiebaCustomDataset(csv_file, img_dir, word2id, max_length=32)
     dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
     print("词表大小:", len(word2id))  # 输出词表大小
     for i, batch in enumerate(dataloader):
         if i == 1:  # 第二个 batch
             print("input_ids[0]:", batch['input_ids'][0])
             print("attention_mask[0]:", batch['attention_mask'][0])
             print("images[0]", batch['image'][0])
             break

