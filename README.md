# BETTER

## To create venv
python -m venv .venv

## Activate the venv
.venv/Scripts/activate

## Download requierments 
pip install --no-cache-dir -r requirements.txt

## To run the server locally
uvicorn app:app --reload

## User Folder Structure

```
d----- users
|       rw---- users.json
|       d----- uid
|               rw---- models.json
|               rw---- current_model.json
|               d----- model_id
|                       d----- similarity
|                               rw---- dendrogram.json
|                               rw---- edges_df.csv
|                               rw---- logistic_regression_model.json
|                       d----- dissimilarity
|                               rw---- dendrogram.json
|                               rw---- edges_df.csv
|                               rw---- logistic_regression_model.json
|                       d----- count
|                               rw---- dendrogram.json
|                               rw---- edges_df.csv
|                               rw---- logistic_regression_model.json
```


## API

### NMA
- **Endpoint**: `http://127.0.0.1:8000/nma/`
- **Methods**: `POST`


```bash
{
    "model_file": "file.keras",
    "dataset": "imagenet",
    "graph_type": "similarity",
    "model_id": None,
    "min_confidence": 0.8,
    "top_k": 4
}
```

### Dataset's labels
- **Endpoint**: `http://127.0.0.1:8000/datasets/{dataset_name}/labels`
- **Methods**: `GET`


### Dendrogram

- **Endpoint**: `http://127.0.0.1:8000/dendrograms`
- **Methods**: `POST`

```bash
{
    "model_id": "uuid",
    "graph_type":  "count",
    "selected_labels": ["Persian_cat", "tabby", "Madagascar_cat", "Egyptian_cat", "pug", "boxer", "Norwich_terrier", "kuvasz", "minivan"]
}
```

### Change Cluster Name

- **Endpoint**: `http://127.0.0.1:8000//dendrograms/auto_naming`
- **Methods**: `PUT`

```bash
{
    "model_id": "uuid",
    "graph_type":  "count",
    "selected_labels": ["Persian_cat", "tabby", "Madagascar_cat", "Egyptian_cat", "pug", "boxer", "Norwich_terrier", "kuvasz", "minivan"],
    "cluster_id": 1263,
    "new_name": "Cats"
}
```

### Whitebox Testing
- **Endpoint**: `http://127.0.0.1:8001/whitebox_testing/`
- **Methods**: `POST`


```bash
{
    "model_id": "uuid",
    "graph_type":  "count",
    "source_labels": ["Persian_cat", "tabby", "Madagascar_cat", "Egyptian_cat"],
    "target_labels": ["pug", "boxer", "Norwich_terrier", "kuvasz"]
}
```

### Query - Hierarchy check
- **Endpoint**: `http://127.0.0.1:8000/query`
- **Methods**: `POST`


```bash
curl -X POST "http://127.0.0.1:8000/query" \
-F "model_filename=cifar100_resnet" \
-F "dataset=cifar100" \
-F "image=@path/to/your/image.jpg" \
-F "dendrogram_filename=edges_dissimilarity_cifar100"
```

### Adversarial
## Adversarial modal generation
- **Endpoint**: `http://127.0.0.1:8000/adversarial/generate`
- **Methods**: `POST`

> **Note:** The `clean_images` and `adversarial_images_images` fields are optional. You can include zero, one, or multiple files for each, for better results advised to provide at least 40-60 examples each.

**With images:**
```bash
curl -X POST "http://127.0.0.1:8000/adversarial/generate" \
-F "current_model_id=uuid" \
-F "graph_type=similarity" \
-F "clean_images=@img1.npy" \
-F "clean_images=@img2.npy" \
-F "adversarial_images_images=@adv1.npy" \
-F "adversarial_images_images=@adv2.npy"
```

**Without images:**
```bash
curl -X POST "http://127.0.0.1:8000/adversarial/generate" \
-F "current_model_id=uuid" \
-F "graph_type=similarity"
```

## Adversarial image detection
- **Endpoint**: `http://127.0.0.1:8000/adversarial/detect`
- **Methods**: `POST`

> **Need to have generated modal for this**

```bash
curl -X POST "http://127.0.0.1:8000/adversarial/detect" \
-F "current_model_id=uuid" \
-F "graph_type=similarity" \
-F "image=@path/to/your/image.jpg" 
```

## Modal Analysis
- **Endpoint**: `http://127.0.0.1:8000/adversarial/analyze`
- **Methods**: `POST`
```bash
curl -X POST "http://127.0.0.1:8000/adversarial/detect" \
-F "current_model_id=uuid" \
-F "graph_type=similarity" \
-F "image=@path/to/your/image.jpg" 
-F "attack_type=pgd"
```
