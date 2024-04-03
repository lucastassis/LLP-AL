import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_mlp(model, data_loader, device):
    y_pred = []
    y_true = []
    # eval
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_pred += pred.argmax(dim=1).cpu().tolist()
            y_true += y.cpu()
    # compute metrics
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="macro")
    re = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    return (acc, pr, re, f1)

def eval_dllp(model, data_loader, device):
    y_pred = []
    y_true = []
    # eval
    model.eval()
    with torch.no_grad():
        for X, y, _, _ in data_loader:
            X, y = X[0].to(device), y[0].to(device)
            _, pred = model(X)
            y_pred += pred.argmax(dim=1).cpu().tolist()
            y_true += y.cpu()
    # compute metrics
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="macro")
    re = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    return (acc, pr, re, f1)

