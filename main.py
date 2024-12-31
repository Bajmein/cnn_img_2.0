from torch import device, cuda
from torch.utils.data import DataLoader
from src.entrenamiento import entrenar_modelo
from src.modelo import ModeloCNN
from src.carga_datos import CargaDatos

def main() -> None:
    # TODO: Luego, agregar y llamar desde el archivo yaml las configuraciones.
    # Configuraciones
    ruta_datos: str = './data'
    tamano_imagen: int = 224
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 0.001

    # Crear instancia de CargaDatos
    cargador: CargaDatos = CargaDatos(ruta=ruta_datos, tamano_imagen=tamano_imagen, batch_size=batch_size)
    dataloader: DataLoader = cargador.cargar_datos()
    print(f'Datos cargados correctamente: {len(dataloader)} batch(es) en total.')

    # Crear instancia del modelo
    dispositivo: device = device('cuda' if cuda.is_available() else 'cpu')
    nombre_dispositivo: str = cuda.get_device_name(cuda.current_device()) if cuda.is_available() else 'CPU'

    modelo: ModeloCNN = ModeloCNN().to(dispositivo)
    print(f'Modelo cargado en dispositivo: {dispositivo}')
    print(f'Nombre del dispositivo: {nombre_dispositivo}')


    # Entrenar modelo
    entrenar_modelo(modelo, dataloader, dispositivo=dispositivo, num_epochs=num_epochs, lr=learning_rate)

if __name__ == '__main__':
    try:
        print('Inicio del programa')
        main()

    except KeyboardInterrupt:
        print('Programa interrumpido')
