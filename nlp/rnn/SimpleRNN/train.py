import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from basic import ImdbDataset, SimpleRnnModel

warehouse_dir = "../../../warehouse"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = ImdbDataset(
    os.path.join(warehouse_dir, "./nlp/aclImdb"),
    "../../tokenizer/english/tokenizer.json",
    32,
    True,
)
dl = DataLoader(dataset=ds, batch_size=128, shuffle=True, drop_last=True)

model = SimpleRnnModel(32768, 32, 32).to(device)
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
    for batch_X, batch_y in dl:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = loss_func(outputs.squeeze(), batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")