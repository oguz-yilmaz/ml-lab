## Açıklama
Makine Öğrenmesi (BLM5110) dersi kapsamında TensorFlow ve Keras kullanılarak geliştirilen
sınıflandırma modeli.

## Gereksinimler
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows

# Install requirements
pip install -r requirements.txt

# Run the code
python model.py
```

## Çalıştırma
Eğitim:
python train.py

Değerlendirme:
python eval.py

### Dosya Düzeni
proje_klasoru/
 |- train.py
 |- eval.py
 |- requirements.txt
 |- dataset/
 |- results/
