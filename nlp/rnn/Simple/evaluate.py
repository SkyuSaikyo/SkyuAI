import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from basic import ImdbDataset, SimpleModel

warehouse_dir = "../../../warehouse"

embedding_dim = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

res = []
res.append(["epoch\\w"] + [epoch for epoch in range(4, 65, 4)])

for w in [16, 32, 64, 128, 256]:
    print(f"loading train dataset, w={w}")
    train_ds = ImdbDataset(
        os.path.join(warehouse_dir, "./nlp/aclImdb"),
        "../../tokenizer/english/tokenizer.json",
        w,
        True,
    )
    train_dl = DataLoader(
        dataset=train_ds, batch_size=128, shuffle=True, drop_last=True
    )

    print(f"loading test dataset, w={w}")
    test_ds = ImdbDataset(
        os.path.join(warehouse_dir, "./nlp/aclImdb"),
        "../../tokenizer/english/tokenizer.json",
        w,
        False,
    )
    test_dl = DataLoader(dataset=test_ds, batch_size=128, shuffle=True)

    model = SimpleModel(32768, embedding_dim, w).to(device)

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_res = [w]

    for epoch in range(1, 65):
        for train_inputs, train_labels in train_dl:
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(
                device
            )
            outputs = model(train_inputs)
            loss = loss_func(outputs.squeeze(), train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 4 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for test_inputs, test_labels in test_dl:
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(
                        device
                    )
                    outputs = model(test_inputs)
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
                accuracy = f"{(correct / total * 100):.2f}"
                print(f"w={w}, epoch={epoch}, accuracy={accuracy}")
                epoch_res.append(accuracy)
            model.train()
    res.append(epoch_res)


csv_file = f"evaluation embedding_dim={embedding_dim}.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows([[row[i] for row in res] for i in range(len(res[0]))])
