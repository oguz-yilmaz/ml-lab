## Açıklama
Makine Öğrenmesi (BLM5110) dersi kapsamında Numpy kullanılarak geliştirilen
sınıflandırma modeli.

## Gereksinimler
```bash
python -m venv venv

source venv/bin/activate  # For Mac/Linux
# venv\Scripts\activate  # For Windows

pip install -r requirements.txt

```

## Çalıştırma
```python
# Modeli oluşturmak için
python train.py

# Modeli değerlendirmek için
python eval.py

# Modeli grafiklerini goruntulemek için
python plot.py
```

* `plot.py` ile olusturulan tum grafikleri `plots/` klasorunde bulabilirsiniz

### Dosya Düzeni
proje_klasoru/
 |- train.py
 |- eval.py
 |- plot.py
 |- model.pkl
 |- requirements.txt
 |- dataset/
 |- plots/
