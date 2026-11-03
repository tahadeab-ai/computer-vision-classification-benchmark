import pytest
import torch
from PIL import Image
from cv_benchmark.data.transforms import get_transforms


def test_custom_cnn_transforms():
    tf_train = get_transforms(model_name="custom_cnn", is_train=True)
    tf_eval = get_transforms(model_name="custom_cnn", is_train=False)

    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    t_train = tf_train(img)
    t_eval = tf_eval(img)

    assert isinstance(t_train, torch.Tensor)
    assert t_train.shape == (3, 32, 32)
    assert t_eval.shape == (3, 32, 32)


def test_pretrained_transforms():
    tf_train = get_transforms(model_name="resnet50", is_train=True, image_size=(224, 224))
    tf_eval = get_transforms(model_name="resnet50", is_train=False, image_size=(224, 224))

    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    t_train = tf_train(img)
    t_eval = tf_eval(img)

    assert t_train.shape == (3, 224, 224)
    assert t_eval.shape == (3, 224, 224)
