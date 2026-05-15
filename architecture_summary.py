import torch
from model import PhishingMultimodalNet


def analyze_architecture():
    print("Ładowanie modelu do analizy (to może potrwać kilka sekund)...")
    model = PhishingMultimodalNet()

    print("\n" + "=" * 70)
    print(f"{'RAPORT ARCHITEKTURY: MULTIMODAL PHISHING DETECTOR':^70}")
    print("=" * 70)


    print(f"{'Moduł Sieci':<25} | {'Wszystkie Parametry':<22} | {'Trenowalne Parametry':<20}")
    print("-" * 70)

    # 1. Analiza gałęzi wizyjnej (ResNet18)
    vis_params = sum(p.numel() for p in model.vision.parameters())
    vis_train = sum(p.numel() for p in model.vision.parameters() if p.requires_grad)
    print(f"{'ResNet18 (Wzrok)':<25} | {vis_params:>22,} | {vis_train:>20,}")

    # 2. Analiza gałęzi tekstowej (DistilBERT)
    txt_params = sum(p.numel() for p in model.text.parameters())
    txt_train = sum(p.numel() for p in model.text.parameters() if p.requires_grad)
    print(f"{'DistilBERT (Tekst)':<25} | {txt_params:>22,} | {txt_train:>20,}")

    # 3. Analiza klasyfikatora końcowego (Fuzja)
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    cls_train = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"{'Klasyfikator (Fuzja)':<25} | {cls_params:>22,} | {cls_train:>20,}")

    print("-" * 70)

    # 4. Podsumowanie
    tot_params = sum(p.numel() for p in model.parameters())
    tot_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'SUMA CAŁKOWITA':<25} | {tot_params:>22,} | {tot_train:>20,}")
    print("=" * 70)


if __name__ == "__main__":
    analyze_architecture()