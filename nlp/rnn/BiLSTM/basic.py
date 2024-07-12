import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, state_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, state_dim, batch_first=True, bidirectional=True
        )
        # Since LSTM is bidirectional, the output dimension of LSTM will be 2 * state_dim
        # Adjust the input size of the linear layer accordingly
        self.fc = nn.Linear(2 * state_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # LSTM outputs a tuple (output, (h_n, c_n))
        # output: [batch_size, seq_len, 2 * state_dim]
        # h_n: [num_layers * num_directions, batch_size, state_dim]
        _, (h_n, _) = self.lstm(x)

        # Concatenate the final states of the forward and backward LSTMs
        # h_n is stacked as [forward_1, backward_1, forward_2, backward_2, ..., forward_layers, backward_layers]
        # We want the final hidden states of the forward and backward LSTMs
        # So, take h_n[-2,:,:] (last one of the forward LSTM) and h_n[-1,:,:] (last one of the backward LSTM)
        concatenated_h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        # Apply the fully connected layer
        x = self.fc(concatenated_h)
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
