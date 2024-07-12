import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from basic import ImdbDataset, LSTMModel

warehouse_dir = "../../../warehouse"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

ds = ImdbDataset(
    os.path.join(warehouse_dir, "./nlp/aclImdb"),
    "../../tokenizer/english/tokenizer.json",
    64,
    True,
)
dl = DataLoader(dataset=ds, batch_size=128, shuffle=True, drop_last=True)

model = LSTMModel(32768, 32, 32).to(device)
model_save_path = "./model.pth"

loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

start_epoch = 0
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch")

for epoch in range(start_epoch, num_epochs):
    for inputs, labels in dl:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")
