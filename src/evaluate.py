from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            predictions.extend(outputs.argmax(dim=1).tolist())
            targets.extend(labels.tolist())
    return accuracy_score(targets, predictions)
