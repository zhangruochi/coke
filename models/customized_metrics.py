import torch

def compute_accuracy(model, data_loader, device):
    corrected_pred, num_examples = 0,0
    model.eval()
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        predicted_labels = torch.argmax(probas, dim=1)
        num_examples += targets.size(0)
        corrected_pred += torch.sum((predicted_labels == targets))
    
    return float(corrected_pred) / num_examples