```bash
conda env create -n medical python=3.10
```
```bash
pip install .
```
```bash
mlflow server --host 0.0.0.0 --port 2000
export MLFLOW_TRACKING_URI=http://localhost:2000
```

```bash
cd training 
python download.py
```
```bash
python train.py
```