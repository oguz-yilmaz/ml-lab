## Description

A classification model developed using Numpy for the Machine Learning (BLM5110) course.

## Requirements

```bash
python -m venv venv

source venv/bin/activate  # For Mac/Linux
# venv\Scripts\activate  # For Windows

pip install -r requirements.txt
```

## Running the Code

```python
# To create the model
python train.py

# To evaluate the model
python eval.py

# To visualize the model's plots
python plot.py
```

* You can find all plots created by `plot.py` in the `plots/` directory.

### File Structure

```bash
project_folder/
 ├─ train.py
 ├─ eval.py
 ├─ plot.py
 ├─ model.pkl
 ├─ requirements.txt
 ├─ dataset/
 └─ plots/
```
