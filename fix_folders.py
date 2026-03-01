import os

folders = [
    "models/research",
    "data/raw",
    "data/processed",
    "data/samples",
    "notebooks",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Klasörün içine boş bir .gitkeep dosyası oluştur
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass
    print(f".gitkeep oluşturuldu: {folder}")