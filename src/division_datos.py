import os
import random
import shutil

class DivisionDatos:
    def __init__(self, ruta_origen, ruta_destino, proporciones=(0.7, 0.15, 0.15)):
        self.ruta_origen = ruta_origen
        self.ruta_destino = ruta_destino
        self.proporciones = proporciones

    def dividir(self):
        os.makedirs(self.ruta_destino, exist_ok=True)
        for split in ["entrenamiento", "evaluacion", "prueba"]:
            os.makedirs(os.path.join(self.ruta_destino, split), exist_ok=True)

        clases = os.listdir(self.ruta_origen)
        for clase in clases:
            ruta_clase = os.path.join(self.ruta_origen, clase)
            imagenes = os.listdir(ruta_clase)
            random.shuffle(imagenes)

            n_total = len(imagenes)
            n_train = int(n_total * self.proporciones[0])
            n_val = int(n_total * self.proporciones[1])

            splits = {
                "entrenamiento": imagenes[:n_train],
                "evaluacion": imagenes[n_train:n_train + n_val],
                "prueba": imagenes[n_train + n_val:]
            }

            for split, imagenes_split in splits.items():
                split_path = os.path.join(self.ruta_destino, split, clase)
                os.makedirs(split_path, exist_ok=True)
                for imagen in imagenes_split:
                    shutil.copy(os.path.join(ruta_clase, imagen), os.path.join(split_path, imagen))
