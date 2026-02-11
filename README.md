# multi-task-learning-usai-2026

##  🤖 Install and Set Up Env For  Unlearn Model 

### 🐍 [1]-Create Conda Environment 
- [1.1]-Python **3.6.9** and Create Conda Environment
```
conda create -n bitnet-unlearn-env python=3.6.9 -y
conda activate bitnet-unlearn-env 
```
- [1.2]-Add Virtual Environment to Jupyter Notebook
```
pip install --user ipykernel
python -m ipykernel install --user --name=bitnet-unlearn-env 
```

### 📦 [2]-Required Packages

```
pip install \
tensorflow==2.6.0rc0 \
keras==2.6.0rc0 \
numpy==1.19.5 \
scipy==1.4.1 \
pandas==1.1.5 \
matplotlib==3.3.4 \
opencv-python==4.6.0.66 \
scikit-learn==0.23.0 \
torch==1.10.1 \
torchvision==0.11.2 \
scikit-image==0.17.2 \
Pillow==8.4.0
```
Key libraries and versions:

| Package | Version |
|------|--------|
| tensorflow | 2.6.0rc0 |
| keras | 2.6.0rc0 |
| numpy | 1.19.5 |
| scipy | 1.4.1 |
| pandas | 1.1.5 |
| matplotlib | 3.3.4 |
| opencv-python | 4.6.0.66 |
| scikit-learn | 0.23.0 |
| torch | 1.10.1 |
| torchvision | 0.11.2 |
| scikit-image | 0.17.2 |
| Pillow | 8.4.0 |
| seaborn | - |

```
conda install seaborn
```
