import torch

def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_checkpoint(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model
