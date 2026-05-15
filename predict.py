import torch
import os
import random
from torchvision import transforms
from PIL import Image
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer
import torch.nn.functional as F


from model import PhishingMultimodalNet


def extract_text_from_html(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text[:512]
    except Exception:
        return ""


def analyze_folder(folder_path, model, tokenizer, device, transform):
    img_path = os.path.join(folder_path, 'shot.png')
    txt_path = os.path.join(folder_path, 'html.txt')

    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print(f"BŁĄD: W folderze {folder_path} brakuje plików. Pomijam.")
        return

    # Przetwarzanie obrazu
    try:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = 50000000

        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except Exception:
        # Zabezpieczenie (czarne tło)
        image = torch.zeros((1, 3, 224, 224)).to(device)

    # Przetwarzanie tekstu
    raw_text = extract_text_from_html(txt_path)
    tokens = tokenizer(
        raw_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    # Predykcja
    with torch.no_grad():
        outputs = model(image, tokens['input_ids'], tokens['attention_mask'])
        probabilities = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)

    # Wynik
    class_names = ['BEZPIECZNA', 'PHISHING']
    result = class_names[predicted.item()]
    confidence = conf.item() * 100

    print(f"Ścieżka: {folder_path}")
    print(f"WERDYKT: {result}")
    print(f"PEWNOŚĆ: {confidence:.2f}%")
    print(f"--------------------------------------------------")


def get_random_folder(base_dir):
    if not os.path.exists(base_dir):
        return None
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return random.choice(folders) if folders else None


if __name__ == "__main__":
    print("Ładowanie modelu...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicjalizacja komponentów
    model = PhishingMultimodalNet().to(device)
    try:
        model.load_state_dict(torch.load('phishing_multimodal_model.pth', weights_only=True))
    except FileNotFoundError:
        print("BŁĄD: Brak pliku z wagami modelu! Upewnij się, że masz plik .pth w tym folderze.")
        exit()

    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Losowanie ścieżek
    phishing_folder = get_random_folder(os.path.join('dataset', 'phishing'))
    benign_folder = get_random_folder(os.path.join('dataset', 'benign'))

    print("\n--------------------ROZPOCZĘCIE ANALIZY--------------------")

    if phishing_folder:
        print("\n[TEST 1: STRONA Z FOLDERU PHISHING]")
        analyze_folder(phishing_folder, model, tokenizer, device, transform)
    else:
        print("Brak danych w folderze dataset/phishing.")

    if benign_folder:
        print("\n[TEST 2: STRONA Z FOLDERU BENIGN]")
        analyze_folder(benign_folder, model, tokenizer, device, transform)
    else:
        print("Brak danych w folderze dataset/benign.")