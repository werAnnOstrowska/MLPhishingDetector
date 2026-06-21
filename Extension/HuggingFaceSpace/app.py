from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image, ImageFile
from bs4 import BeautifulSoup
import base64, io
from model import PhishingMultimodalNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 50_000_000

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhishingMultimodalNet().to(device)
model.load_state_dict(torch.load(
    'phishing_multimodal_model.pth',
    map_location=device,
    weights_only=True
))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)[:512]


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json

    # — obraz —
    try:
        img_bytes = base64.b64decode(data['screenshot'])
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception:
        image_tensor = torch.zeros((1, 3, 224, 224)).to(device)

    # — tekst —
    raw_text = extract_text(data.get('html', ''))
    tokens = tokenizer(
        raw_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    # — predykcja —
    with torch.no_grad():
        outputs = model(image_tensor, tokens['input_ids'], tokens['attention_mask'])
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    return jsonify({
        "verdict": "PHISHING" if predicted.item() == 1 else "SAFE",
        "confidence": round(conf.item() * 100, 2)
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)  