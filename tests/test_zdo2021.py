import pytest
import os
import skimage.io
import glob
import numpy as np
from pathlib import Path
import zdo2021.main

# cd ZDO2021
# python -m pytest

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


    assert prediction.shape[0] == imgs.shape[0]


    # Toto se bude spouštět všude mimo GitHub
    if not os.getenv('CI'):
        import matplotlib.pyplot as plt
        plt.imshow(prediction[0])
        plt.show()


    # import json
    # gt_ann = json.loads(Path(dataset_path)/"annotations/instances_default.json")
    # assert f1score(ground_true_masks, prediction) > 0.55


def f1score(gt_ann, prediction):
    pass

def prepare_ground_true_masks(gt_ann, filname):
    pass
