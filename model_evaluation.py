import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset and prints the accuracy.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch_geometric.loader.DataLoader): DataLoader containing test data.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.

    Returns:
        None: Prints the test accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    y_true = []  # List to store true labels
    y_pred = []  # List to store predicted labels

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        # Iterate through the batches in the test_loader
        for batch in test_loader:
            batch = batch.to(device)  # Move the batch to the correct device
            out = model(batch)  # Get model predictions
            _, predicted = torch.max(out, 1)  # Get the predicted class (index with highest score)
            y_true.extend(batch.y.tolist())  # Append true labels to y_true list
            y_pred.extend(predicted.tolist())  # Append predicted labels to y_pred list
    
    # Calculate accuracy using sklearn's accuracy_score
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")  # Print the accuracy percentage
