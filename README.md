# BETTER

## To create venv
python -m venv .venv

## Activate the venv
.venv/Scripts/activate

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

### Hierarchical Clustering - Confusion Matrix

- **Endpoint**: `http://127.0.0.1:8001/hierarchical_clusters/confusion_matrix`
- **Methods**: `POST`

```bash
{
    "model_filename": "mini_imagenet",
    "edges_df_filename": "edges_dissimilarity_cifar100",
    "dataset": "imagenet"
}
```

### Hierarchical Clustering - Confusion Matrix

- **Endpoint**: `http://127.0.0.1:8000/hierarchical_clusters/sub_hierarchical_clusters`
- **Methods**: `POST`

```bash
{
    "dataset": "imagenet",
    "selected_labels": ["Persian_cat", "tabby", "orange", "lemon", "zucchini", "broccoli", "teapot", "coffeepot", "warplane", "space_shuttle", "American_coot", "black_swan"],
    "z_filename": "dendrogram_similarity_mini_imagenet"
}
```


### Whitebox Testing
- **Endpoint**: `http://127.0.0.1:8001/whitebox_testing/`
- **Methods**: `POST`


```bash
{
    "model_filename":"cifar100_resnet",
    "source_labels":["forest", "maple_tree", "oak_tree", "willow_tree", "pine_tree", "palm_tree"],
    "target_labels":["girl", "boy", "woman", "man","baby"],
    "edges_data_filename": "edges_dissimilarity_cifar100"
}
```