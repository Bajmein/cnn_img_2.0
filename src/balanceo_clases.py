import os
import random
from torchvision import transforms
# from PIL import Image

class BalanceoClases:
    def __init__(self, ruta_origen: str) -> None:
        self.ruta_origen = ruta_origen
        self.augmentaciones = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ])

    def augmentar_clase(self, clase: str, n_augmentaciones: int) -> None:
        ruta_clase: str = os.path.join(str(self.ruta_origen), str(clase))
        imagenes: list[str] = os.listdir(ruta_clase)

        for i in range(n_augmentaciones):
            ruta_imagen = random.choice(imagenes)
            # imagen = Image.open(os.path.join(ruta_clase, ruta_imagen))
            # imagen_aumentada = self.augmentaciones(imagen)
            # nombre_aumento = f"augment_{i}_{os.path.basename(ruta_imagen)}"
            # imagen_aumentada.save(os.path.join(ruta_clase, nombre_aumento))

            # Solo imprime el nombre del aumento sin guardarlo
            print(f"Imagen aumentada de {os.path.basename(ruta_imagen)}: augment_{i}_{os.path.basename(ruta_imagen)}")

    def balancear(self):
        clases: list[str] | list[bytes] = os.listdir(self.ruta_origen)
        conteo: dict[str, int] = {clase: len(os.listdir(os.path.join(self.ruta_origen, clase))) for clase in clases}
        max_clase: int = max(conteo.values())

        for clase, tamano in conteo.items():
            if tamano < max_clase:
                augmentos_necesarios: int = max_clase - tamano
                print(f"Augmentando clase '{clase}' con {augmentos_necesarios} imágenes adicionales...")
                self.augmentar_clase(clase, augmentos_necesarios)


if __name__ == "__main__":
    # Prueba
    ruta = '../data'
    instancia = BalanceoClases(ruta_origen=ruta)
    instancia.balancear()