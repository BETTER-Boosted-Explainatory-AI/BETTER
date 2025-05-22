# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import tensorflow_io as tfio
# import tensorflow as tf
# from services.models_service import _load_model
# from utilss.enums.datasets_enum import DatasetsEnum

# dataset_cfg = {
#     "top_k": 4,
#     "min_confidence": 0.8,
#     "dataset": DatasetsEnum.CIFAR100.value,
# }

# model_key = (
#     "92457494-3061-700c-8e14-f1ab249392d7/"
#     "33b8e5d1-c7d2-4c35-8a29-5764e8dd86fd/"
#     "cifar100_resnet.keras"
# )                       

# m = _load_model(DatasetsEnum.CIFAR100.value, model_key, dataset_cfg)
