import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer


Image.MAX_IMAGE_PIXELS = 50000000

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultimodalPhishingDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_text_length=512):
        self.root_dir = root_dir
        self.transform = transform
        self.max_text_length = max_text_length
        self.samples = []

        for label, category in enumerate(['benign', 'phishing']):
            category_dir = os.path.join(root_dir, category)
            if not os.path.exists(category_dir):
                print(f"Ostrzeżenie: Brak folderu {category_dir}")
                continue

            for folder_name in os.listdir(category_dir):
                folder_path = os.path.join(category_dir, folder_name)
                if os.path.isdir(folder_path):
                    self.samples.append((folder_path, label))

    def __len__(self):
        return len(self.samples)

    def _extract_text(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                return text[:self.max_text_length]
        except Exception:
            return ""

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]

        img_path = os.path.join(folder_path, 'shot.png')
        txt_path = os.path.join(folder_path, 'html.txt')


        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:

            image = torch.zeros((3, 224, 224))

        # Wyciąganie tekstu
        text = self._extract_text(txt_path)

        return image, text, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MultimodalPhishingDataset(root_dir='dataset', transform=transform)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    print(f"Znaleziono próbek: {len(dataset)}")

    if len(dataset) > 0:
        img, raw_text, label = dataset[0]
        tokenized = tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        print(f"Kształt obrazu: {img.shape}")
        print(f"Kształt tokenów tekstu (input_ids): {tokenized['input_ids'].shape}")
        print("Mamy to. Gotowe do treningu!")
    else:
        print("BŁĄD: Nie znaleziono danych w folderze 'dataset'.")