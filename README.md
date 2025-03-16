# BETTER

## To create venv
python -m venv .venv

## Activate the venv
venv/Scripts/activate

## Download requierments 
pip install --no-cache-dir -r requirements.txt

## To run the server locally
uvicorn app:app --reload

## API

### Hierarchical Clustering
- **Endpoint**: `http://127.0.0.1:8001/hierarchical_clusters/`
- **Methods**: `POST`


```bash
{
    "model_filename": "mini_imagenet",
    "graph_type": "similarity",
    "dataset": "imagenet"
}
```