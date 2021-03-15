import pytest
import os
import skimage.io
import glob
import numpy as np
import zdo2021.main


def test_run_random():
    vdd = zdo2021.main.VarroaDetector()

    # Nastavte si v operačním systém proměnnou prostředí 'VARROA_DATA_PATH' s cestou k datasetu
    dataset_path = os.getenv('VARROA_DATA_PATH')

    # print(f'dataset_path = {dataset_path}')
    files = glob.glob(f'{dataset_path}/images/*.jpg')
    cislo_obrazku = np.random.randint(0, len(files))
    filename = files[cislo_obrazku]

    im = skimage.io.imread(filename)
    imgs = np.expand_dims(im, axis=0)
    print(f"imgs.shape={imgs.shape}")

    prediction = vdd.predict(imgs)
    assert prediction.shape[0] == imgs.shape[0]
