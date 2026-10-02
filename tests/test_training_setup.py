"""Training setup regression tests; compiled rotated NMS is outside this suite.

Run from the repository root: python -m pytest tests/test_training_setup.py
The fail-fast extension stub lets these tests exercise real plotting/model code
without a platform-specific CUDA build. It must never perform NMS.
"""
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image, ImageFont
import pytest
import torch
import yaml


OBB_ROOT = Path(__file__).resolve().parents[1] / 'shrimp_OBB'


@pytest.fixture(scope='module')
def plots():
    extension = types.ModuleType('utils.nms_rotated.nms_rotated_ext')

    def unavailable(*args, **kwargs):
        raise AssertionError('These tests must not call compiled NMS')

    extension.nms_rotated = extension.nms_poly = unavailable
    sys.path.insert(0, str(OBB_ROOT))
    try:
        with patch.dict(sys.modules, {extension.__name__: extension}):
            yield importlib.import_module('utils.plots')
    finally:
        sys.path.remove(str(OBB_ROOT))


@pytest.fixture(autouse=True)
def model_directory(plots, monkeypatch, tmp_path):
    monkeypatch.setattr(plots, 'MODEL_DIR', tmp_path)
    monkeypatch.setattr(plots, 'check_font', lambda **kwargs: ImageFont.load_default())
    plots._load_regression_model.cache_clear()
    yield tmp_path
    plots._load_regression_model.cache_clear()


class ScaleModel:
    def __init__(self, scale):
        self.scale = scale

    def predict(self, features):
        return np.asarray(features).sum(axis=1) * self.scale


def test_import_without_measurement_models(plots, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Importing plots must not load a checkpoint')

    monkeypatch.setattr(joblib, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    importlib.reload(plots)


@pytest.mark.parametrize('pil', [False, True])
@pytest.mark.parametrize('tensor_list', [False, True])
def test_generic_annotation_without_models(plots, pil, tensor_list):
    annotator = plots.Annotator(np.full((160, 160, 3), 255, dtype=np.uint8), pil=pil, line_width=2)
    poly = [20, 20, 120, 20, 120, 80, 20, 80]
    if tensor_list:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        poly = list(torch.tensor(poly, device=device))  # same form as detect.py's *poly
    annotator.poly_label(poly, 'shrimp 0.9', color=(255, 0, 0))
    result = annotator.result()
    assert result.shape == (160, 160, 3)
    assert (result[20, 60] == [255, 0, 0]).all()
    assert (result[48:75, 70:130] != 255).any()  # class label is drawn
    assert plots._load_regression_model.cache_info().currsize == 0


@pytest.mark.parametrize('prediction', [False, True])
def test_training_and_validation_mosaics_without_models(plots, tmp_path, prediction):
    images = np.full((2, 3, 160, 160), 255, dtype=np.uint8)
    if prediction:
        targets = np.array([[0, 0, 70, 60, 60, 20, 0, 0.9]], dtype=np.float32)
    else:
        targets = np.zeros((1, 187), dtype=np.float32)
        targets[0, :7] = [0, 0, 70, 60, 60, 20, 0]
    destination = tmp_path / 'mosaic.jpg'
    plots.plot_images(images, targets, paths=['sample.jpg', 'empty.jpg'], fname=destination, names=['shrimp'])
    rendered = np.asarray(Image.open(destination))
    assert rendered.shape[0] > 160
    assert (rendered[50:70, 40:100] < 200).any()  # OBB/label survives PIL rendering
    assert plots._load_regression_model.cache_info().currsize == 0


def test_models_load_once_and_predictions_are_preserved(plots, model_directory, monkeypatch, tmp_path):
    filenames = ['final_linear_model_length.pkl', 'final_linear_model_width.pkl',
                 'polynomial_regression_model_degree3.pkl', 'multi_feature_model.pkl']
    for filename, scale in zip(filenames, [2, 3, 0.4, 0.5]):
        joblib.dump(ScaleModel(scale), model_directory / filename)
    real_load = joblib.load
    calls = []

    def record(path):
        calls.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(joblib, 'load', record)
    unrelated = tmp_path / 'other working directory'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    for _ in range(2):
        annotator = plots.Annotator(np.zeros((180, 500, 3), dtype=np.uint8))
        poly = [10, 20, 110, 20, 110, 40, 10, 40]
        annotator.shrimp_label(poly, width=20, label='shrimp')
        assert (annotator.length, annotator.width, annotator.weight) == (80, 24, 52)
        annotator.shrimp_label(poly, width=None, label='shrimp')
        assert annotator.length == 80 and annotator.width is None and annotator.weight == 32
    assert sorted(calls) == sorted(model_directory / name for name in filenames)


def test_missing_model_is_reported_only_when_measurement_requested(plots, model_directory):
    with pytest.raises(FileNotFoundError, match='final_linear_model_length.pkl') as error:
        plots.predict_length(100)
    assert str(model_directory) in str(error.value)
    # Failed loads are not cached; supplying a model later works without restart.
    joblib.dump(ScaleModel(2), model_directory / 'final_linear_model_length.pkl')
    assert plots.predict_length(100) == 200


def test_upstream_hyperparameters_include_obb_settings():
    hyp = yaml.safe_load((OBB_ROOT / 'data/hyps/obb/hyp.finetune_dota.yaml').read_text(encoding='utf-8'))
    assert hyp['cls_theta'] == 180
    assert hyp['theta'] > 0 and hyp['theta_pw'] > 0 and hyp['csl_radius'] > 0
    assert 0 < hyp['lr0'] < 1 and 0 <= hyp['mosaic'] <= 1


def test_dataset_example_matches_loader(plots, tmp_path, monkeypatch):
    from utils.datasets import img2label_paths
    from utils.general import check_dataset

    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load((OBB_ROOT / 'data/bottom_shrimp.example.yaml').read_text(encoding='utf-8'))
    dataset = tmp_path / data['path']
    for split in ('train', 'val'):
        (dataset / split / 'images').mkdir(parents=True)
    result = check_dataset(data)
    assert result['nc'] == len(result['names']) == 1
    image = (Path(result['train']) / 'sample.jpg').resolve()
    assert Path(img2label_paths([str(image)])[0]) == dataset / 'train/labelTxt/sample.txt'


@pytest.mark.parametrize('api_available', [True, False])
def test_missing_weights_use_pinned_release(plots, tmp_path, monkeypatch, api_available):
    from utils import downloads

    urls, requests = [], []

    def release_request(url, timeout):
        requests.append(url)
        assert timeout > 0
        if not api_available:
            raise downloads.requests.ConnectionError('API unavailable')
        return types.SimpleNamespace(raise_for_status=lambda: None,
                                     json=lambda: {'assets': [{'name': 'yolov5n.pt'}], 'tag_name': 'v6.0'})

    def save(file, url, **kwargs):
        urls.append(url)
        assert Path(file).parent.is_dir()
        Path(file).write_bytes(b'checkpoint fixture')

    monkeypatch.setattr(downloads.requests, 'get', release_request)
    monkeypatch.setattr(downloads, 'safe_download', save)
    destination = tmp_path / 'weight/yolov5n.pt'
    assert downloads.attempt_download(destination) == str(destination)
    assert urls == ['https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt']
    assert requests == ['https://api.github.com/repos/ultralytics/yolov5/releases/tags/v6.0']
    downloads.attempt_download(destination)  # existing files must not be fetched again
    assert len(urls) == len(requests) == 1


def test_custom_weight_url_is_preserved(plots, tmp_path, monkeypatch):
    from utils import downloads

    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(downloads, 'safe_download', lambda **kwargs: seen.append(kwargs['url']))
    url = 'https://example.com/custom.pt'
    assert downloads.attempt_download(url) == 'custom.pt'
    assert seen == [url]


def test_legacy_checkpoint_load_overrides_new_torch_default(plots, monkeypatch):
    from utils.torch_utils import load_yolo_checkpoint

    def new_torch_load(path, map_location=None, *, weights_only=True):
        assert weights_only is False
        assert map_location == 'cpu'
        return {'model': 'fixture'}

    monkeypatch.setattr(torch, 'load', new_torch_load)
    assert load_yolo_checkpoint('trusted.pt', map_location='cpu')['model'] == 'fixture'


def test_train_parser_uses_supplied_configuration(plots, monkeypatch):
    import train

    monkeypatch.setattr(sys, 'argv', ['train.py', '--data', 'data/bottom_shrimp.example.yaml'])
    args = train.parse_opt()
    assert str(args.data) == 'data/bottom_shrimp.example.yaml'
    assert Path(args.hyp).is_file()
    assert str(args.weights).replace('\\', '/').endswith('weight/yolov5n.pt')


def test_real_obb_training_batch_without_biometric_models(plots, tmp_path, monkeypatch):
    from models.yolo import Model
    from utils import datasets
    from utils.loss import ComputeLoss

    # Windows-spawned workers cannot inherit the test's extension stub.
    # Keep the real label parser but run cache checks in threads for this test.
    monkeypatch.setattr(datasets, 'Pool', datasets.ThreadPool)
    images, labels = tmp_path / 'images', tmp_path / 'labelTxt'
    images.mkdir()
    labels.mkdir()
    for i in range(2):
        Image.fromarray(np.full((64, 64, 3), 80 + i * 20, dtype=np.uint8)).save(images / f'{i}.png')
        (labels / f'{i}.txt').write_text('10 10 45 10 45 25 10 25 shrimp 0\n')
    hyp = yaml.safe_load((OBB_ROOT / 'data/hyps/obb/hyp.finetune_dota.yaml').read_text(encoding='utf-8'))
    hyp['label_smoothing'] = 0.0
    data = datasets.LoadImagesAndLabels(str(images), ['shrimp'], img_size=64, batch_size=2, hyp=hyp, rect=True)
    batch, targets, _, _ = data.collate_fn([data[0], data[1]])
    config = {'nc': 1, 'depth_multiple': 1.0, 'width_multiple': 1.0,
              'anchors': [[10, 13, 16, 30, 33, 23]],
              'backbone': [[-1, 1, 'Conv', [16, 3, 2]], [-1, 1, 'Conv', [32, 3, 2]]],
              'head': [[[1], 1, 'Detect', ['nc', 'anchors']]]}
    model = Model(config, ch=3, nc=1).train()
    model.hyp = hyp
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    loss, components = ComputeLoss(model)(model(batch.float() / 255), targets)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    assert not torch.equal(parameter, before)
    assert plots._load_regression_model.cache_info().currsize == 0
