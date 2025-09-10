import torch
from PIL import Image
from torchvision import transforms
import jieba
import os
from glob import glob
from Image_Text_Enconder import ImageEncoder, MyTextEncoder
from dataloader_image_text import build_vocab
from const import device


# ======================
# 图像预处理（和训练保持一致）
# ======================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ======================
# 文本预处理（jieba 分词）
# ======================
def preprocess_text(text, word2id, max_length=32):
    words = [w for w in jieba.cut(text) if w.strip()]
    input_ids = [word2id.get(w, word2id["[PAD]"]) for w in words]

    # 加上特殊 token
    input_ids = [word2id["[CLS]"]] + input_ids[:max_length - 2] + [word2id["[SEP]"]]

    # padding
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [word2id["[PAD]"]] * pad_len

    attention_mask = [1 if id != word2id["[PAD]"] else 0 for id in input_ids]

    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0), \
           torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)


# ======================
# 检索函数
# ======================
def encode_image(image_path, image_encoder):
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = image_encoder(img)
    return feat


def encode_text(text, text_encoder, word2id):
    input_ids, attn_mask = preprocess_text(text, word2id)
    input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
    with torch.no_grad():
        feat = text_encoder(input_ids, attn_mask)
    return feat


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 加载词表
    word2id, id2word = build_vocab("./dataset/ImageWordData.csv")

    # 创建模型并加载权重
    vocab_size = len(word2id)
    image_encoder = ImageEncoder(embed_dim=512).to(device)
    text_encoder = MyTextEncoder(vocab_size=vocab_size, embed_dim=512).to(device)

    image_encoder.load_state_dict(torch.load("image_encoder_weights.pth", weights_only=True))
    text_encoder.load_state_dict(torch.load("text_encoder_weights.pth", weights_only=True))

    image_encoder.eval()
    text_encoder.eval()

    # 例子：用图片检索文字
    img_feat = encode_image("./dataset/ImageData/Image14001001-0000.jpg", image_encoder)
    txt_feat = encode_text("《绿色北京》摄影大赛胡子<人名>作品", text_encoder, word2id)
    sim = cosine_similarity(img_feat, txt_feat)
    print("相似度:", sim.item())

    # 例子：用文字检索图片
    query_text = "《绿色北京》摄影大赛胡子<人名>作品"  # 你想检索的文本
    txt_feat = encode_text(query_text, text_encoder, word2id)
    image_folder = "./dataset/ImageData"
    image_paths = glob(os.path.join(image_folder, "*.jpg"))

    similarities = []
    for path in image_paths:
        img_feat = encode_image(path, image_encoder)
        sim = cosine_similarity(txt_feat, img_feat).item()
        similarities.append((path, sim))

    # 按相似度排序，取前 K 个
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_k = 5
    print(f"\n【文字检索图片】 查询: {query_text}")
    for i, (path, score) in enumerate(similarities[:top_k]):
        print(f"Top {i+1}: {path}, 相似度={score:.4f}")

