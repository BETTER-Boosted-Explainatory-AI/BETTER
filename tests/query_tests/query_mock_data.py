user = {
    "user_id": "8482bdbf-fada-4a24-a45d-15ff1498b519",
    "email": "test@example.com",
    "password": "TestP@ssw0rd"
}

models_metadata = {
    "models": [
        {
            "model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
            "file_name": "resnet50_imagenet.keras",
            "dataset": "imagenet",
            "graph_type": [
                "similarity"
            ],
            "min_confidence": 0.8,
            "top_k": 4,
            "batch_jobs": [
                {
                    "job_id": "1f84cd45-c969-4b86-8eee-eb05f3d87ed4",
                    "job_graph_type": "similarity",
                    "job_status": "succeeded",
                    "timestamp": "2025-05-24T17:16:11.419177"
                }
            ]
        }
    ]
}

model_id = "35f658ac-aa29-461e-85fe-f7dcfe638dde"

labels = ["Persian_cat", "tabby", "Egyptian_cat", "bathtub", "tub"]

mock_dendrogram = {
    "id": 1003,
    "name": "whole_121",
    "children": [
        {
            "id": 1001,
            "name": "whole_116",
            "children": [
                {
                    "id": 283,
                    "name": "Persian_cat"
                },
                {
                    "id": 1000,
                    "name": "bathtub_2",
                    "children": [
                        {
                            "id": 435,
                            "name": "bathtub"
                        },
                        {
                            "id": 876,
                            "name": "tub"
                        }
                    ],
                    "value": 0.0
                }
            ],
            "value": 0.7073672218248248
        },
        {
            "id": 1002,
            "name": "domestic_cat",
            "children": [
                {
                    "id": 281,
                    "name": "tabby"
                },
                {
                    "id": 285,
                    "name": "Egyptian_cat"
                }
            ],
            "value": 0.7376661213347688
        }
    ],
    "value": 1.2287538811527459
}

mock_sub_dendrogram = {
    "id": 1000,
    "name": "bathtub_2",
    "children": [
        {
            "id": 435,
            "name": "bathtub"
        },
        {
            "id": 876,
            "name": "tub"
        }
    ],
    "value": 0.0
}

top_label = "bathtub"
query_predictions = ["bathtub", "tubs"]

top_k_predictions = [("bathtub", 0.99), ("tubs", 0.01)]
verbal_explanation = ["bathtub", "tubs", "bathtub_2", "whole_116", "whole_121"]
