import torch
import torch.nn.functional as F
from const import device

def contrastive_loss(image_features, text_features, temperature=0.07):
    # 相似度矩阵
    logits = torch.matmul(image_features, text_features.T) / temperature  # [B, B]
    labels = torch.arange(len(image_features)).to(image_features.device)  # 正样本对角线

    # 双向 loss（图像->文本 + 文本->图像）
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


@torch.no_grad()
def compute_accuracy(image_encoder, text_encoder, dataloader):
    image_encoder.eval()
    text_encoder.eval()

    all_image_feats = []
    all_text_feats = []

    for batch in dataloader:
        images, input_ids, attention_mask = batch['image'], batch['input_ids'], batch['attention_mask']
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        image_features = image_encoder(images)   # [B, D]
        text_features = text_encoder(input_ids, attention_mask)  # [B, D]

        all_image_feats.append(image_features)
        all_text_feats.append(text_features)

    image_feats = torch.cat(all_image_feats, dim=0)   # [N, D]
    text_feats = torch.cat(all_text_feats, dim=0)     # [N, D]

    # 相似度矩阵 [N, N]
    sim_matrix = torch.matmul(image_feats, text_feats.T)

    # 图像 -> 文本 检索精度
    preds_i2t = sim_matrix.argmax(dim=1)  # 每张图最相似的文本
    labels = torch.arange(len(image_feats)).to(device)
    acc_i2t = (preds_i2t == labels).float().mean().item()

    # 文本 -> 图像 检索精度
    preds_t2i = sim_matrix.argmax(dim=0)  # 每段文本最相似的图
    acc_t2i = (preds_t2i == labels).float().mean().item()

    return (acc_i2t + acc_t2i) / 2

