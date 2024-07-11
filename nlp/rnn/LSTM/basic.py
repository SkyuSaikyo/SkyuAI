import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, state_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, state_dim, batch_first=True)
        self.fc = nn.Linear(state_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n)
        x = self.sigmoid(x)
        return x


class ImdbDataset(Dataset):
    def __init__(
        self, data_dir: str, tokenizer_dir: str, max_length: int, is_train: bool
    ):
        self.data = []
        self.max_length = max_length
        tokenizer = Tokenizer.from_file(tokenizer_dir)
        target_paths = [
            os.path.join(data_dir, f'{"train" if is_train else "test"}/pos'),
            os.path.join(data_dir, f'{"train" if is_train else "test"}/neg'),
        ]
        for label, target_path in enumerate(target_paths):
            for file_name in os.listdir(target_path):
                with open(
                    os.path.join(target_path, file_name), mode="r", encoding="utf8"
                ) as file:
                    self.data.append((tokenizer.encode(file.read().lower()).ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self._aligner(self.data[idx][0]),
            torch.tensor(self.data[idx][1], dtype=torch.float32),
        )

    def _aligner(self, sentence: list) -> torch.Tensor:
        sentence_tensor = torch.tensor(sentence, dtype=torch.long)
        if len(sentence_tensor) > self.max_length:
            sentence_tensor = sentence_tensor[: self.max_length]
        if len(sentence_tensor) < self.max_length:
            padding = torch.zeros(
                self.max_length - len(sentence_tensor), dtype=torch.long
            )
            sentence_tensor = torch.cat([sentence_tensor, padding])
        return sentence_tensor
