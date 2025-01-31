import os
import torch
from torch.utils.data import DataLoader

from basic import ImdbDataset, SimpleModel

warehouse_dir = "../../../warehouse"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

ds = ImdbDataset(
    os.path.join(warehouse_dir, "./nlp/aclImdb"),
    "../../tokenizer/english/tokenizer.json",
    256,
    False,
)
dl = DataLoader(dataset=ds, batch_size=128)

model = SimpleModel(32768, 8, 256).to(device)
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0

print("Testing the model...")
with torch.no_grad():
    for inputs, labels in dl:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
