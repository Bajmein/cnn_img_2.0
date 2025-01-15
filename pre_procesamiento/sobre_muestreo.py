import yaml
import pandas as pd
import os
from imblearn.over_sampling import RandomOverSampler
from typing import Self, Any
from src.carga_datos import CreadorDataset
from collections import Counter
from time import sleep
import shutil


class Sobremuestreo:
    def __init__(self: Self, ruta_entrenamiento: str, ruta_validacion: str, ruta_prueba: str) -> None:
        self.ruta_entrenamiento = f'../{ruta_entrenamiento}'
        self.ruta_validacion = f'../{ruta_validacion}'
        self.ruta_prueba = f'../{ruta_prueba}'

    def aplicar_sobremuestreo(self: Self) -> Any:
        df_entrenamiento, _, _ = CreadorDataset.etiquetar_dataset(
            self.ruta_entrenamiento,
            self.ruta_validacion,
            self.ruta_prueba
        )

        X = df_entrenamiento['ruta_archivo']
        y = df_entrenamiento['etiqueta']

        ros = RandomOverSampler(sampling_strategy='minority')
        X_res, y_res = ros.fit_resample(pd.DataFrame(X), pd.Series(y))

        print(f"Balance de clases después del sobremuestreo: {dict(Counter(y_res))}")

        for ruta, etiqueta in zip(X_res['ruta_archivo'], y_res):
            categoria = 'normal' if etiqueta == 0 else 'pneumonia'

            if etiqueta == 0:
                nombre_nuevo = f"{os.path.splitext(os.path.basename(ruta))[0]}_sampled{os.path.splitext(ruta)[1]}"
                ruta_destino = os.path.join(self.ruta_entrenamiento, categoria, nombre_nuevo)
                shutil.copy(ruta, ruta_destino)

if __name__ == '__main__':
    ruta_config: str = '../config/config.yaml'

    with open(ruta_config, 'r') as archivo:
        configuraciones = yaml.safe_load(archivo)
        print('\nConfiguraciones cargadas exitosamente.')
        sleep(1.5)

    train_dir = configuraciones['directorios']['entrenamiento']
    val_dir = configuraciones['directorios']['validacion']
    test_dir = configuraciones['directorios']['prueba']

    on = Sobremuestreo(train_dir, val_dir, test_dir)
    on.aplicar_sobremuestreo()