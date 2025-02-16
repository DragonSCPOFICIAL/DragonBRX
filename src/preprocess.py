import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import json

class QADataset(Dataset):
    def __init__(self, questions, answers, vocab):
        self.questions = questions
        self.answers = answers
        self.vocab = vocab

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Tokenização e criação de vocabulário
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, row in data.iterrows():
        for word in row["question"].split() + row["answer"].split():
            if word not in vocab:
                vocab[word] = len(vocab)

    # Salvar vocabulário
    with open("dataset/vocab.json", "w") as f:
        json.dump(vocab, f)

    # Codificar perguntas e respostas
    questions = [[vocab.get(word, vocab["<UNK>"]) for word in q.split()] for q in data["question"]]
    answers = [[vocab.get(word, vocab["<UNK>"]) for word in a.split()] for a in data["answer"]]

    # Criar DataLoader
    dataset = QADataset(questions, answers, vocab)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return train_loader, vocab
