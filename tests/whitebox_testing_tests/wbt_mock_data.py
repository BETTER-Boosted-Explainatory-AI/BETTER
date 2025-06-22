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
source_labels = ["bathtub", "bath tub"]
target_labels = ["Persian_cat"]

mock_edges_df = """image_id,source,target,target_probability
2838,Persian_cat,bathtub,0.040728264
2838,Persian_cat,tub,0.022001969
2838,Persian_cat,window_screen,0.013135839
2839,Persian_cat,Pomeranian,0.012880797
2839,Persian_cat,tabby,0.0016715704
2839,Persian_cat,lynx,0.0010061789
2845,Siamese_cat,Egyptian_cat,0.009015327
2845,Siamese_cat,remote_control,0.0006121531
2845,Siamese_cat,radiator,0.00024338275"""