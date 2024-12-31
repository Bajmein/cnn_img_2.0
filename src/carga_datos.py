from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Self


class CargaDatos:
    def __init__(self: Self, ruta: str, tamano_imagen: int, batch_size: int) -> None:
        self.ruta_datos: str = ruta
        self.tamano_imagen: int = tamano_imagen
        self.batch_size: int = batch_size

    def aplicar_transformaciones(self: Self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.tamano_imagen, self.tamano_imagen)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def cargar_datos(self: Self) -> DataLoader:
        transformaciones: transforms.Compose = self.aplicar_transformaciones()
        dataset: datasets.ImageFolder = datasets.ImageFolder(root=self.ruta_datos, transform=transformaciones)
        dataloader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader