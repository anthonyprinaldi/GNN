from torch_geometric.datasets import Planetoid
import lightning.pytorch as L
import torch.nn as nn
import torch
import torch.nn.functional as F
from GNN import GNN

EPOCHS = 200
LR = 1e-3
WD = 1e-4

if __name__ == "__main__":
    dataset = Planetoid(root='data/', name='PubMed')
    data = dataset[0]
    model = GNN(
        dataset.num_features,
        [256, 128, 32], 
        dataset.num_classes
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay=WD)
    metric = nn.NLLLoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(data)
        loss = metric(preds[data.train_mask], data.y[data.train_mask])     
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch} | loss: {loss.item():6.4f}')

        if epoch % 5  == 0:
            model.eval()
            with torch.no_grad():
                preds = model(data)
                loss = metric(preds[data.val_mask], data.y[data.val_mask])
                preds = preds.argmax(dim=-1)
                acc = (preds[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
                print(f'    Epoch: {epoch} | val_loss: {loss.item():6.4f} | val_acc: {acc.item():6.4f}')
            model.train()

    # evaluate model
    model.eval()
    preds = model(data)
    loss = metric(preds[data.test_mask], data.y[data.test_mask])
    preds = preds.argmax(dim = -1)
    acc = (preds[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    print(f'Epoch: {epoch} | test_loss: {loss.item():6.4f} | test_acc: {acc.item():6.4f}')

    