import torch
import  string
punctuation = string.punctuation + "，。！？；：”“‘’（）【】《》、—…· ' ' '\t'"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")