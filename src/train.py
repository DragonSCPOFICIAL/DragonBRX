import torch
from model import TransformerModel
from preprocess import load_data, preprocess_data
from checkpoint import save_checkpoint

def train_model():
    # Carregar e pré-processar dados
    data = load_data("dataset/qa_data.csv")
    train_loader, vocab = preprocess_data(data)

    # Definir modelo
    model = TransformerModel(vocab_size=len(vocab), embed_dim=512, num_heads=8, num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Treinamento
    for epoch in range(10):
        for batch in train_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        save_checkpoint(model, f"models/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train_model()
