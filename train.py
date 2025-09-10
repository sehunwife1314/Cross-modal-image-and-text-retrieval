import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy import public
from Image_Text_Enconder import ImageEncoder, MyTextEncoder
from dataloader_image_text import JiebaCustomDataset, build_vocab
from loss_acc import contrastive_loss, compute_accuracy
from torch.utils.data import DataLoader
from const import device


def train(dataloader, image_encoder, text_encoder, optimizer, num_epochs=10):
    train_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()

        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images, input_ids, attention_mask = batch['image'], batch['input_ids'], batch['attention_mask']
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 🔍 第一次 batch 时打印 GPU 使用情况
            if epoch == 0 and batch_idx == 0:
                check_gpu_usage(
                    model_dict={"image_encoder": image_encoder, "text_encoder": text_encoder},
                    batch_dict={"images": images, "input_ids": input_ids, "attention_mask": attention_mask}
                )

            # 前向
            image_features = image_encoder(images)
            text_features = text_encoder(input_ids, attention_mask)
            loss = contrastive_loss(image_features, text_features)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        acc = compute_accuracy(image_encoder, text_encoder, dataloader)

        train_losses.append(avg_loss)
        accuracies.append(acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    return train_losses, accuracies



def plot_training_curves(losses, accuracies):
    """
    losses: 每个 epoch 的平均训练损失
    accuracies: 每个 epoch 的检索精度
    """
    num_epochs = len(losses)
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12,5))

    # Loss 曲线
    plt.subplot(1,2,1)
    plt.plot(epochs, losses, marker='o', color='blue', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracies, marker='o', color='orange', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Retrieval Accuracy Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def check_gpu_usage(model_dict, batch_dict):
    print("======== GPU 调用检查 ========")
    print("当前可用 CUDA:", torch.cuda.is_available())
    print("当前设备:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        print("GPU 名称:", torch.cuda.get_device_name(0))
        print("显存占用: {:.2f} GB".format(torch.cuda.memory_allocated(0)/1024**3))
        print("显存缓存: {:.2f} GB".format(torch.cuda.memory_reserved(0)/1024**3))

    print("\n--- 模型所在设备 ---")
    for name, model in model_dict.items():
        print(f"{name}: {next(model.parameters()).device}")

    print("\n--- Batch 数据所在设备 ---")
    for name, tensor in batch_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tensor.device}, shape={tuple(tensor.shape)}")

    print("==============================\n")

# 用法示例：
# 在你拿到 batch 的地方（比如训练循环里第一轮），调用一次：


if __name__ == "__main__":

    csv_file = "./dataset/ImageWordData.csv"
    img_dir = "./dataset/ImageData"
    word2id, id2word = build_vocab(csv_file=csv_file)
    dataset = JiebaCustomDataset(csv_file, img_dir, word2id, max_length=32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    image_encoder = ImageEncoder(embed_dim=512).to(device)
    text_encoder = MyTextEncoder(vocab_size=6450, embed_dim=512).to(device)
    params = list(image_encoder.parameters()) + list(text_encoder.parameters())
    optimizer = optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
    num_epochs = 10
    losses, accs = train(dataloader, image_encoder, text_encoder, optimizer, num_epochs)
    plot_training_curves(losses, accs)
    torch.save(image_encoder.state_dict(), "image_encoder_weights.pth")
    torch.save(text_encoder.state_dict(), "text_encoder_weights.pth")
    print("模型保存完成！")
