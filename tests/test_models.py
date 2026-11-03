import pytest
import torch
from cv_benchmark.models import CustomCNN, create_model


def test_custom_cnn_forward():
    model = CustomCNN(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


@pytest.mark.parametrize("model_name", ["mobilenet_v2", "resnet50"])
def test_pretrained_models_feature_extraction(model_name):
    # Test with pretrained=False to enable fast shape testing without network download
    model, head_names = create_model(model_name=model_name, num_classes=100, pretrained=False, mode="feature_extraction")
    
    # Check that backbone parameters are frozen
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_params:
        assert any(h in name for h in head_names)

    # Check forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 100)


def test_pretrained_fine_tuning_unfrozen():
    model, _ = create_model(model_name="mobilenet_v2", num_classes=100, pretrained=False, mode="fine_tuning")
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable is True
