import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from utilss.classes.cifar100 import Cifar100
from utilss.classes.imagenet import ImageNet

@pytest.fixture
def cifar100():
    dataset = Cifar100()
    dataset.load("cifar100")
    return dataset

@pytest.fixture
def imagenet():
    dataset = ImageNet()
    dataset.load("imagenet")
    return dataset

def test_cifar100_load(cifar100):
    assert cifar100.x_train is not None, "x_train should not be None"
    assert cifar100.y_train is not None, "y_train should not be None"
    assert cifar100.x_test is not None, "x_test should not be None"
    assert cifar100.y_test is not None, "y_test should not be None"

    assert cifar100.x_train.shape == (50000, 32, 32, 3), f"x_train shape should be (50000, 32, 32, 3) but is {cifar100.x_train.shape}"
    assert cifar100.y_train.shape == (50000,), f"y_train shape should be (50000,) but is {cifar100.y_train.shape}"
    assert cifar100.x_test.shape == (10000, 32, 32, 3), f"x_test shape should be (10000, 32, 32, 3) but is {cifar100.x_test.shape}"
    assert cifar100.y_test.shape == (10000,), f"y_test shape should be (10000,) but is {cifar100.y_test.shape}"

def test_imagenet_load(imagenet):
    assert len(imagenet.train_data) > 0, "Train data should not be empty"
    assert len(imagenet.test_data) > 0, "Test data should not be empty"

    assert len(imagenet.train_data) == 50000, "Train data should be 50000"
    assert len(imagenet.test_data) == 10000, "Test data should be 10000"

def test_cifar100_get_train_image_by_id(cifar100):
    image_id = 1
    image, label = cifar100.get_train_image_by_id(image_id)
    assert image is not None, "Image should not be None"
    assert label is not None, "Label should not be None"

def test_cifar100_get_test_image_by_id(cifar100):
    image_id = 1
    image, label = cifar100.get_test_image_by_id(image_id)
    assert image is not None, "Image should not be None"
    assert label is not None, "Label should not be None"


def test_imagenet_get_train_image_by_id(imagenet):
    image_id = 1
    image, label = imagenet.get_train_image_by_id(image_id)
    assert image is not None, "Image should not be None"
    assert label is not None, "Label should not be None"

def test_imagenet_get_test_image_by_id(imagenet):
    image_id = 1
    image, label = imagenet.get_test_image_by_id(image_id)
    assert image is not None, "Image should not be None"
    assert label is not None, "Label should not be None"