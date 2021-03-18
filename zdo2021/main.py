import numpy as np
from . import podpurne_funkce

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, data):
        # output = np.zeros_like(data)
        output = [{"image_id": 1, "bbox": [1280.2, 840.71, 26.3, 22.87]},
                  {"image_id": 1, "bbox": [1636.9, 503.2, 22.82, 25.11]},
                  {"image_id": 2, "bbox": [671.61, 653.11, 14.15, 15.85]},
                  {"image_id": 2, "bbox": [58.33, 428.42, 17.67, 12.45]}
                  ]
        return output