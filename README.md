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
python training/download.py
```
```bash
python training/train.py
```
```bash
docker compose build --no-cache app   
```
```bash
python versioning.py
```