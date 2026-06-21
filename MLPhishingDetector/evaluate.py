import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import DistilBertTokenizer
import random


from dataset_loader import MultimodalPhishingDataset
from model import PhishingMultimodalNet


def plot_learning_curves():
    print("Generowanie krzywych uczenia...")

    epochs = [1, 2, 3, 4, 5, 6]
    train_loss = [0.1568, 0.1575, 0.2708, 0.0109, 0.2856, 0.0417]
    val_loss = [0.1984, 0.1655, 0.1728, 0.1423, 0.1673, 0.1690]
    val_acc = [92.75, 93.80, 93.10, 95.05, 93.85, 93.95]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Wykres Loss
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color='tab:red', label='Train Loss')
    ax1.plot(epochs, val_loss, marker='s', linestyle='--', color='tab:orange', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Wykres Accuracy na drugiej osi Y
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(epochs, val_acc, marker='^', linestyle='-', color='tab:blue', label='Validation Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    # Zaznaczenie momentu Early Stopping (Najlepszy model)
    plt.axvline(x=4, color='green', linestyle=':', label='Early Stopping (Zapis)')

    plt.title('Krzywe Uczenia - Multimodal Phishing Detector')
    plt.tight_layout()
    plt.savefig('wykres_uczenia.png', dpi=300)
    print("Zapisano: wykres_uczenia.png")


def unnormalize_image(tensor):
    """Cofa normalizację PyTorcha, żeby matplotlib mógł pokazać normalny obrazek"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def evaluate_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUruchamianie ewaluacji na: {device}")

    # 1. Ładowanie danych
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = MultimodalPhishingDataset(root_dir='dataset', transform=transform)

    # Wybieramy 1000 losowych próbek
    indices = random.sample(range(len(full_dataset)), min(1000, len(full_dataset)))
    test_dataset = Subset(full_dataset, indices)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Ładowanie Twojego wytrenowanego modelu
    model = PhishingMultimodalNet().to(device)
    try:
        model.load_state_dict(torch.load('phishing_multimodal_model.pth', weights_only=True))
        print("Pomyślnie załadowano wagi modelu z pliku .pth!")
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku phishing_multimodal_model.pth. Przerwij operację.")
        return

    model.eval()

    all_preds = []
    all_labels = []


    true_positives = []
    mistakes = []

    print("Analizowanie próbek. Proszę czekać...")
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors='pt').to(
                device)

            outputs = model(images, tokens['input_ids'], tokens['attention_mask'])
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


            for i in range(len(preds)):
                actual = labels[i].item()
                predicted = preds[i].item()
                img_tensor = images[i]


                if actual == 1 and predicted == 1 and len(true_positives) < 3:
                    true_positives.append((img_tensor, actual, predicted))


                if actual != predicted and len(mistakes) < 1:
                    mistakes.append((img_tensor, actual, predicted))

    # Macierz pomyłek
    print("\nGenerowanie Macierzy Pomyłek...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bezpieczne', 'Phishing'],
                yticklabels=['Bezpieczne', 'Phishing'])
    plt.ylabel('Rzeczywistość')
    plt.xlabel('Predykcja Modelu')
    plt.title('Macierz Pomyłek (Confusion Matrix)')
    plt.tight_layout()
    plt.savefig('macierz_pomylek.png', dpi=300)
    print("Zapisano: macierz_pomylek.png")

    # wizualizacja
    print("\nGenerowanie wizualizacji wyników...")
    samples_to_show = true_positives + mistakes

    if len(samples_to_show) > 0:
        fig, axes = plt.subplots(1, len(samples_to_show), figsize=(15, 4))
        if len(samples_to_show) == 1:
            axes = [axes]

        classes = ['Bezpieczna', 'PHISHING']

        for idx, (img_tensor, actual, pred) in enumerate(samples_to_show):
            img_np = unnormalize_image(img_tensor)
            axes[idx].imshow(img_np)
            axes[idx].axis('off')

            #
            color = 'green' if actual == pred else 'red'
            title = f"Prawda: {classes[actual]}\nModel: {classes[pred]}"
            axes[idx].set_title(title, color=color, fontweight='bold')

        plt.suptitle("Przykładowe analizy stron (Po lewej poprawne, po prawej błąd)", fontsize=14)
        plt.tight_layout()
        plt.savefig('przyklady_analizy.png', dpi=300)
        print("Zapisano: przyklady_analizy.png")

    print("\nZAKOŃCZONO.")


if __name__ == "__main__":
    plot_learning_curves()
    evaluate_and_visualize()