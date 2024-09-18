import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            _, predicted = torch.max(out, 1)
            y_true.extend(batch.y.tolist())
            y_pred.extend(predicted.tolist())
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
