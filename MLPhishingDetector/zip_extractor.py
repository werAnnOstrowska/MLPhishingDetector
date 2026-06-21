import zipfile
import os
import random


def extract_random_samples(zip_path, dest_path, num_samples=5000):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print(f"\n--- Otwieranie archiwum: {zip_path} ---")
    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            all_files = archive.namelist()

            folders = list(set([f.split('/')[0] for f in all_files if '/' in f]))

            selected_folders = random.sample(folders, min(num_samples, len(folders)))
            print(f"Znaleziono {len(folders)} unikalnych stron. Wylosowano {len(selected_folders)} do wypakowania.")

            extracted_count = 0
            for folder in selected_folders:
                for item in all_files:

                    if item.startswith(folder + '/'):

                        target_dir = os.path.join(dest_path, folder)

                        if item.endswith('.png'):
                            file_data = archive.read(item)
                            os.makedirs(target_dir, exist_ok=True)
                            safe_path = os.path.join(target_dir, 'shot.png')
                            with open(safe_path, 'wb') as f:
                                f.write(file_data)

                        elif 'html' in item.lower() or item.endswith('.txt'):
                            #dodatkowe bezpieczeńśtwo - zapis jako.txt
                            file_data = archive.read(item)
                            os.makedirs(target_dir, exist_ok=True)
                            safe_path = os.path.join(target_dir, 'html.txt')
                            with open(safe_path, 'wb') as f:
                                f.write(file_data)

                extracted_count += 1
                if extracted_count % 500 == 0:
                    print(f"Wypakowano {extracted_count} z {len(selected_folders)}...")

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {zip_path}. Upewnij się, że leży w tym samym folderze co skrypt.")


#URUCHOMIENIE
print("Rozpoczynam przygotowywanie datasetu...")
extract_random_samples('phish_sample_30k.zip', 'dataset/phishing', num_samples=5000)
extract_random_samples('benign_sample_30k.zip', 'dataset/benign', num_samples=5000)
print("\nOperacja zakończona sukcesem. Gotowe do treningu!")