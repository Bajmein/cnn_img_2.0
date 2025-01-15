import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Self
from colorama import Fore, Style


class CreadorDataset:
    @staticmethod
    def crear_dataset(ruta_carpeta: str) -> pd.DataFrame:
        lista_imagenes: list[list[str]] = []

        for categoria in ['normal', 'pneumonia']:
            ruta_categoria = os.path.join(ruta_carpeta, categoria)

            for nombre_archivo in os.listdir(ruta_categoria):
                ruta_archivo = os.path.join(ruta_categoria, nombre_archivo)
                lista_imagenes.append([ruta_archivo, categoria])

        return pd.DataFrame(lista_imagenes, columns=['ruta_archivo', 'etiqueta'])

    @staticmethod
    def etiquetar_dataset(train_dir: str, val_dir: str, test_dir: str) -> tuple:
        df_entrenamiento: pd.DataFrame = CreadorDataset.crear_dataset(train_dir)
        df_validacion: pd.DataFrame = CreadorDataset.crear_dataset(val_dir)
        df_prueba: pd.DataFrame = CreadorDataset.crear_dataset(test_dir)

        mapeo_etiquetas: dict[str, int] = {'normal': 0, 'pneumonia': 1}
        df_entrenamiento['etiqueta'] = df_entrenamiento['etiqueta'].map(mapeo_etiquetas)
        df_validacion['etiqueta'] = df_validacion['etiqueta'].map(mapeo_etiquetas)
        df_prueba['etiqueta'] = df_prueba['etiqueta'].map(mapeo_etiquetas)

        return df_entrenamiento, df_validacion, df_prueba


class DatasetImagenes(Dataset):
    def __init__(self: Self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self: Self) -> int:
        return len(self.dataframe)

    def __getitem__(self: Self, idx) -> tuple:
        ruta_img = self.dataframe.iloc[idx, 0]
        etiqueta = int(self.dataframe.iloc[idx, 1])
        img: Image = Image.open(ruta_img).convert('L')

        if self.transform:
            img = self.transform(img)

        etiqueta = torch.tensor(etiqueta, dtype=torch.long)
        return img, etiqueta


class PrepararDatos:
    def __init__(self: Self, tamano_imagen: int) -> None:
        self.transform_entrenamiento = transforms.Compose([
            transforms.Resize((tamano_imagen, tamano_imagen)),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_val_prueba = transforms.Compose([
            transforms.Resize((tamano_imagen, tamano_imagen)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def obtener_datasets(self: Self, train_dir: str, val_dir: str, test_dir: str, batch_size: int) -> tuple:
        df_entrenamiento, df_validacion, df_prueba = CreadorDataset.etiquetar_dataset(train_dir, val_dir, test_dir)

        print("\nDistribución de clases en el conjunto de entrenamiento:")
        print(f'{Fore.WHITE}{df_entrenamiento['etiqueta'].value_counts()}{Style.RESET_ALL}')

        print("\nDistribución de clases en el conjunto de validación:")
        print(f'{Fore.WHITE}{df_validacion['etiqueta'].value_counts()}{Style.RESET_ALL}')

        print("\nDistribución de clases en el conjunto de prueba:")
        print(f'{Fore.WHITE}{df_prueba['etiqueta'].value_counts()}{Style.RESET_ALL}')

        dataset_entrenamiento = DatasetImagenes(df_entrenamiento, transform=self.transform_entrenamiento)
        dataset_validacion = DatasetImagenes(df_validacion, transform=self.transform_val_prueba)
        dataset_prueba = DatasetImagenes(df_prueba, transform=self.transform_val_prueba)

        loader_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=batch_size, shuffle=True)
        loader_validacion = DataLoader(dataset_validacion, batch_size=batch_size, shuffle=False)
        loader_prueba = DataLoader(dataset_prueba, batch_size=batch_size, shuffle=False)

        return loader_entrenamiento, loader_validacion, loader_prueba
