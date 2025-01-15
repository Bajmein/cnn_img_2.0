import os
import random
import shutil
from typing import Self
from tqdm import tqdm


class DivisionDatos:
    def __init__(self: Self, ruta_origen: str, ruta_destino: str, proporciones: tuple) -> None:
        self.ruta_origen = ruta_origen
        self.ruta_destino = ruta_destino
        self.proporciones = proporciones

    def dividir(self: Self) -> None:
        os.makedirs(self.ruta_destino, exist_ok=True)

        for split in ["entrenamiento", "evaluacion", "prueba"]:
            os.makedirs(os.path.join(self.ruta_destino, split), exist_ok=True)

        clases: list[str] = os.listdir(self.ruta_origen)

        for clase in tqdm(clases, desc="Dividiendo clases"):
            ruta_clase: str = os.path.join(self.ruta_origen, clase)
            imagenes: list[str] = os.listdir(ruta_clase)
            random.shuffle(imagenes)

            n_total: int = len(imagenes)
            n_entrenamiento: int = int(n_total * self.proporciones[0])
            n_evaluacion: int = int(n_total * self.proporciones[1])

            splits: dict[str, list[str]] = {
                "entrenamiento": imagenes[:n_entrenamiento],
                "evaluacion": imagenes[n_entrenamiento:n_entrenamiento + n_evaluacion],
                "prueba": imagenes[n_entrenamiento + n_evaluacion:]
            }

            for split, imagenes_split in splits.items():
                split_path: str = os.path.join(self.ruta_destino, split, clase)
                os.makedirs(split_path, exist_ok=True)

                for imagen in imagenes_split:
                    shutil.copy(os.path.join(ruta_clase, imagen), os.path.join(split_path, imagen))


if __name__ == "__main__":
    ruta_origen = '../datos/datos_sin_separar'
    ruta_destino = '../datos/datos_separados'
    proporciones = (0.7, 0.15, 0.15)
    DivisionDatos(ruta_origen, ruta_destino, proporciones).dividir()