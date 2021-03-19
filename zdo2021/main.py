import numpy as np
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
from . import podpurne_funkce

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, data):
        """
        :param data: np.ndarray with shape [pocet_obrazku, vyska, sirka, barevne_kanaly]
        :return: shape [pocet_obrazku, vyska, sirka], 0 - nic, 1 - varroa destructor
        """
        print("ahoj")
        output = np.zeros_like(data)
        return output