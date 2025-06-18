"""
tests/test_datasets.py – self-contained unit-tests for Cifar-100 & ImageNet
"""

import os
import sys
import json
import time
import pickle         
import unittest
from io import StringIO
from typing import Dict, List

import dotenv
import numpy as np
from unittest.mock import patch, MagicMock

dotenv.load_dotenv()

# ─────────────────────────── app imports ────────────────────────────
from services.dataset_service import get_dataset_labels
from utilss.classes.datasets.cifar100 import Cifar100
from utilss.classes.datasets.imagenet import ImageNet
from utilss.classes.datasets.dataset_factory import DatasetFactory
from request_models.dataset_model import DatasetLabelsResult


# ╔══════════════════════════════════════════════════════════════════╗
# ║                            TEST CASES                           ║
# ╚══════════════════════════════════════════════════════════════════╝
class TestDatasetInfo(unittest.TestCase):
    """Verify dataset information for CIFAR-100 and (Mini) ImageNet."""

    # ───── common fixtures ──────────────────────────────────────────
    def setUp(self) -> None:
        # Expected sizes
        self.cifar100_expected_labels = 100
        self.cifar100_expected_train_images = 50_000
        self.cifar100_expected_test_images = 10_000

        self.imagenet_expected_labels = 1_000
        self.imagenet_expected_train_images = 10_000

        # Mock CIFAR-100 raw batches
        self.mock_cifar100_train_data = {
            b"data": np.random.randint(0, 255, (50_000, 3072), dtype=np.uint8),
            b"fine_labels": np.random.randint(0, 100, 50_000).tolist(),
        }
        self.mock_cifar100_test_data = {
            b"data": np.random.randint(0, 255, (10_000, 3072), dtype=np.uint8),
            b"fine_labels": np.random.randint(0, 100, 10_000).tolist(),
        }

        # Mock dataset-config files
        self.mock_cifar100_config = {
            "dataset": "cifar100",
            "threshold": 0.5,
            "infinity": 100,
            "labels": [f"class_{i}" for i in range(100)],
        }
        self.mock_imagenet_config = {
            "dataset": "imagenet",
            "threshold": 0.5,
            "infinity": 1000,
            "labels": [f"n{str(i).zfill(8)}" for i in range(1000)],
            "directory_to_readable": {
                f"n{str(i).zfill(8)}": f"class_{i}" for i in range(1000)
            },
        }

    # ──────── simple “service layer” tests (remain unchanged) ───────
    @patch("services.dataset_service.get_dataset_config")
    def test_cifar100_labels_count(self, mock_cfg):
        mock_cfg.return_value = self.mock_cifar100_config
        labels = get_dataset_labels("cifar100")
        self.assertEqual(len(labels), self.cifar100_expected_labels)
        self.assertEqual(labels, self.mock_cifar100_config["labels"])

    @patch("services.dataset_service.get_dataset_config")
    def test_imagenet_labels_count(self, mock_cfg):
        mock_cfg.return_value = self.mock_imagenet_config
        labels = get_dataset_labels("imagenet")
        self.assertEqual(len(labels), self.imagenet_expected_labels)
        self.assertEqual(labels, self.mock_imagenet_config["labels"])

    # ───── CIFAR-100 tests (patch inside cifar100 module) ────────────
    @patch("utilss.classes.datasets.cifar100.get_dataset_config")
    @patch("utilss.classes.datasets.cifar100.unpickle_from_s3")
    @patch.dict(os.environ, {"S3_DATASETS_BUCKET_NAME": "test-bucket"})
    def test_cifar100_image_count(self, mock_unpickle, mock_cfg):
        mock_cfg.return_value = self.mock_cifar100_config
        mock_unpickle.side_effect = [
            self.mock_cifar100_train_data,
            self.mock_cifar100_test_data,
        ]

        ds = Cifar100()
        ds.load("cifar100")

        self.assertEqual(len(ds.x_train), self.cifar100_expected_train_images)
        self.assertEqual(len(ds.x_test), self.cifar100_expected_test_images)
        self.assertEqual(len(ds.y_train), self.cifar100_expected_train_images)
        self.assertEqual(len(ds.y_test), self.cifar100_expected_test_images)

    @patch("utilss.classes.datasets.cifar100.get_dataset_config")
    @patch("utilss.classes.datasets.cifar100.unpickle_from_s3")
    @patch.dict(os.environ, {"S3_DATASETS_BUCKET_NAME": "test-bucket"})
    def test_cifar100_label_mapping(self, mock_unpickle, mock_cfg):
        mock_cfg.return_value = self.mock_cifar100_config
        mock_unpickle.side_effect = [
            self.mock_cifar100_train_data,
            self.mock_cifar100_test_data,
        ]

        ds = Cifar100()
        ds.load("cifar100")

        self.assertEqual(ds.label_to_class_name(42), "class_42")
        self.assertTrue(all(isinstance(lbl, str) for lbl in ds.y_train))
        self.assertTrue(all(isinstance(lbl, str) for lbl in ds.y_test))

    @patch("utilss.classes.datasets.cifar100.get_dataset_config")
    @patch("utilss.classes.datasets.cifar100.unpickle_from_s3")
    @patch.dict(os.environ, {"S3_DATASETS_BUCKET_NAME": "test-bucket"})
    def test_cifar100_data_shape(self, mock_unpickle, mock_cfg):
        mock_cfg.return_value = self.mock_cifar100_config
        mock_unpickle.side_effect = [
            self.mock_cifar100_train_data,
            self.mock_cifar100_test_data,
        ]

        ds = Cifar100()
        ds.load("cifar100")

        self.assertEqual(ds.x_train.shape, (50_000, 32, 32, 3))
        self.assertEqual(ds.x_test.shape, (10_000, 32, 32, 3))

    # ───── ImageNet tests (patch inside imagenet module) ────────────
    @patch("utilss.classes.datasets.imagenet.get_dataset_config")
    def test_imagenet_label_structure(self, mock_cfg):
        mock_cfg.return_value = self.mock_imagenet_config
        ds = ImageNet()
        self.assertEqual(len(ds.labels), self.imagenet_expected_labels)
        self.assertTrue(all(lbl.startswith("n") for lbl in ds.labels))
        self.assertTrue(all(len(lbl) == 9 for lbl in ds.labels))

    @patch("utilss.classes.datasets.imagenet.get_dataset_config")
    def test_imagenet_label_mapping(self, mock_cfg):
        mock_cfg.return_value = self.mock_imagenet_config
        ds = ImageNet()
        self.assertEqual(
            ds.directory_to_labels_conversion("n00000042"), "class_42"
        )
        self.assertEqual(
            ds.get_label_readable_name("n00000042"), "class_42"
        )

    @patch("utilss.classes.datasets.imagenet.get_dataset_config")
    @patch("utilss.classes.datasets.imagenet.S3ImagenetLoader")
    @patch.dict(os.environ, {"S3_DATASETS_BUCKET_NAME": "test-bucket"})
    def test_imagenet_image_count(self, mock_loader_cls, mock_cfg):
        mock_cfg.return_value = self.mock_imagenet_config

        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.get_imagenet_classes.return_value = [
            f"n{str(i).zfill(8)}" for i in range(1000)
        ]
        mock_loader.get_class_images.side_effect = lambda c: [
            f"{c}/img_{i}.JPEG" for i in range(10)
        ]
        mock_loader.get_image_data.return_value = b"fake"

        with patch("utilss.classes.datasets.imagenet.Image") as mock_img_mod:
            mock_img = MagicMock()
            mock_img.mode = "RGB"
            mock_img.resize.return_value = mock_img
            mock_img_mod.open.return_value = mock_img
            with patch("numpy.array", return_value=np.zeros((224, 224, 3))):
                ds = ImageNet()
                ds.load("imagenet")

        self.assertEqual(mock_loader.get_image_data.call_count, 10_000)

    @patch("utilss.classes.datasets.imagenet.get_dataset_config")
    def test_imagenet_bidirectional_label_mapping(self, mock_cfg):
        mock_cfg.return_value = self.mock_imagenet_config
        ds = ImageNet()

        readable_to_synset = {}
        for synset, readable in ds.directory_to_readable.items():
            self.assertNotIn(readable, readable_to_synset)
            readable_to_synset[readable] = synset
        self.assertEqual(len(readable_to_synset), len(ds.directory_labels))

    # ───── factory & service quick checks ───────────────────────────
    def test_dataset_factory_registration(self):
        avail = DatasetFactory.get_available_datasets()
        self.assertIn("cifar100", avail)
        self.assertIn("imagenet", avail)

    @patch("services.dataset_service.get_dataset_config")
    def test_get_dataset_labels_route_success(self, mock_cfg):
        mock_cfg.return_value = self.mock_cifar100_config
        labels = get_dataset_labels("cifar100")
        result = DatasetLabelsResult(data=labels)
        self.assertEqual(len(result.data), self.cifar100_expected_labels)
        self.assertEqual(result.data, self.mock_cifar100_config["labels"])


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     CLI / helper utilities                      ║
# ╚══════════════════════════════════════════════════════════════════╝
def run_dataset_tests(verbose: bool = True):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDatasetInfo)
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    return runner.run(suite)


def run_specific_test(test_name: str):
    suite = unittest.TestSuite()
    suite.addTest(TestDatasetInfo(test_name))
    return unittest.TextTestRunner(verbosity=2).run(suite)


def generate_test_report() -> str:
    buf = StringIO()
    start = time.time()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(TestDatasetInfo)
    )
    end = time.time()

    report = f"""
DATASET UNIT TEST REPORT
========================
Execution Time: {end - start:.2f}s
Total Tests: {result.testsRun}
Passed: {result.testsRun - len(result.failures) - len(result.errors)}
Failed: {len(result.failures)}
Errors: {len(result.errors)}

Details:
{buf.getvalue()}
"""
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "report":
            print(generate_test_report())
        elif cmd == "specific" and len(sys.argv) > 2:
            run_specific_test(sys.argv[2])
        else:
            print("Usage: python -m tests.test_datasets "
                  "[report | specific <test_name>]")
    else:
        unittest.main()
