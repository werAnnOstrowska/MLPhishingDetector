import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import DistilBertTokenizer
from tqdm import tqdm
import copy


from dataset_loader import MultimodalPhishingDataset
from model import PhishingMultimodalNet


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"--- EarlyStopping: {self.counter}/{self.patience} ---")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- ROZPOCZĘCIE TRENINGU (ZABEZPIECZONY BATCH) ---")

    # 1. Przygotowanie danych
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = MultimodalPhishingDataset(root_dir='dataset', transform=transform)

    # Podział 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = PhishingMultimodalNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(patience=2)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 2. Pętla treningowa
    for epoch in range(10):
        # FAZA TRENINGU
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoka {epoch + 1} [TRENING]")
        for images, texts, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors='pt').to(
                device)

            optimizer.zero_grad()
            outputs = model(images, tokens['input_ids'], tokens['attention_mask'])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # FAZA WALIDACJI
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, texts, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)
                tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors='pt').to(
                    device)

                outputs = model(images, tokens['input_ids'], tokens['attention_mask'])
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = (val_correct.double() / val_size) * 100
        print(f"WALIDACJA | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # Sprawdzamy, czy to najlepszy model do tej pory
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # Decyzja Early Stopping
        early_stopper(epoch_val_loss)
        if early_stopper.early_stop:
            print("--- KONIEC: Brak poprawy na walidacji. Przerywam trening. ---")
            break

    # 3. Zapisanie najlepszej wersji modelu
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'phishing_multimodal_model.pth')
    print(f"Sukces! Najlepsza celność na walidacji: {best_acc:.2f}%")


if __name__ == "__main__":
    train_model()