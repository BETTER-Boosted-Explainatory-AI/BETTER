
import os
import sys 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utilss.enums.datasets_enum import DatasetsEnum
from services.models_service import _load_model
import tensorflow as tf
from dotenv import load_dotenv      

load_dotenv()

dataset_cfg = {
    "top_k": 4,
    "min_confidence": 0.8,
    "dataset": DatasetsEnum.CIFAR100.value,
}

m = _load_model(
        dataset_str=DatasetsEnum.CIFAR100.value,
        model_path="better-xai-users/92457494-3061-700c-8e14-f1ab249392d7/33b8e5d1-c7d2-4c35-8a29-5764e8dd86fd/cifar100_resnet.keras",     
        dataset_config=dataset_cfg,
)

assert isinstance(m.model, tf.keras.Model)
print("✓ Model object looks OK")
