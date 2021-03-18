import pytest
import os
import skimage.io
import glob
import numpy as np
from pathlib import Path
import zdo2021.main


def test_run_random():
    vdd = zdo2021.main.VarroaDetector()

    # Nastavte si v operačním systém proměnnou prostředí 'VARROA_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = os.getenv('VARROA_DATA_PATH_', default=Path(__file__).parent / 'test_dataset/')

    # print(f'dataset_path = {dataset_path}')
    files = glob.glob(f'{dataset_path}/images/*.jpg')
    cislo_obrazku = np.random.randint(0, len(files))
    filename = files[cislo_obrazku]

    im = skimage.io.imread(filename)
    imgs = np.expand_dims(im, axis=0)
    # print(f"imgs.shape={imgs.shape}")
    prediction = vdd.predict(imgs)

    assert "image_id" in prediction[0]
    assert "bbox" in prediction[0]

    # import json
    # gt_ann = json.loads(Path(dataset_path)/"annotations/instances_default.json")
    # assert f1score() > 0.55


def f1score(gt_ann, prediction):
    pass
