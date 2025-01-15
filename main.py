import torch
import yaml
import argparse
import os
from time import sleep
from src.carga_datos import PrepararDatos
from src.modelo import ModeloCNN
from src.entrenamiento import entrenar_modelo


def generar_logs(acc_train: float, acc_val: float,
                 tamano_imagen: int, batch_size: int, num_epochs: int,
                 learning_rate: float, dropout: float, log_archivo: str) -> None:

    from datetime import datetime
    timestamp: str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    log: str = (
        f"{timestamp} | TamanoImagen: {tamano_imagen} | Batch: {batch_size} | "
        f"Epochs: {num_epochs} | LR: {learning_rate:.4f} | DROPOUT: {dropout:.1f} | "
        f"AccEntrenamiento: {acc_train:.4f} | AccValidacion: {acc_val:.4f}\n"
    )

    with open(log_archivo, "a") as archivo:
        archivo.write(log)

    print('Log guardado exitosamente.')


def cargar_mejor_modelo(ruta_modelo: str) -> float:
    if os.path.exists(ruta_modelo):
        checkpoint = torch.load(ruta_modelo, weights_only=True)
        return checkpoint.get('best_acc', 0.0)

    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo CNN.")
    parser.add_argument('--apagar', action='store_true', help="Apagar el PC al finalizar.")
    args = parser.parse_args()

    if args.apagar:
        from colorama import Fore, Style
        print(f"\n{Fore.LIGHTRED_EX}Apagando automático activado.{Style.RESET_ALL}")

    try:
        ruta_config: str = 'config/config.yaml'
        with open(ruta_config, 'r') as archivo:
            configuraciones = yaml.safe_load(archivo)
            print('\nConfiguraciones cargadas exitosamente.')
            sleep(1.5)

        train_dir = configuraciones['directorios']['entrenamiento']
        val_dir = configuraciones['directorios']['validacion']
        test_dir = configuraciones['directorios']['prueba']
        ruta_modelo = configuraciones['directorios']['modelo_ok']
        logs_ruta = configuraciones['directorios']['logs']
        tamano_imagen = configuraciones['entrenamiento']['tamano_imagen']
        batch_size = configuraciones['entrenamiento']['tamano_batch']
        num_epochs = configuraciones['entrenamiento']['num_epochs']
        learning_rate = configuraciones['entrenamiento']['taza_aprendizaje']
        dropout = configuraciones['entrenamiento']['dropout']

        dispositivo: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {dispositivo}.")
        sleep(1.5)

        preparar_datos: PrepararDatos = PrepararDatos(tamano_imagen)
        loader_entrenamiento, loader_validacion, loader_prueba = preparar_datos.obtener_datasets(
            train_dir, val_dir, test_dir, batch_size
        )
        print("\nDatos cargados correctamente.")
        sleep(1.5)

        modelo: ModeloCNN = ModeloCNN(dropout=dropout)
        modelo_entrenado, acc_entrenamiento, acc_validacion = entrenar_modelo(
            modelo=modelo,
            train_loader=loader_entrenamiento,
            val_loader=loader_validacion,
            dispositivo=dispositivo,
            num_epochs=num_epochs,
            lr=learning_rate
        )
        print("\nEntrenamiento completado.")

        modelo_actual = cargar_mejor_modelo(ruta_modelo=ruta_modelo)

        if acc_validacion > modelo_actual:
            torch.save({'state_dict': modelo_entrenado.state_dict(), 'best_acc': acc_validacion}, f=ruta_modelo)
            print("Nuevo modelo guardado como 'modelo_entrenado.pth' (mejorado).")

        else:
            print("El modelo no superó el rendimiento anterior. No se guardó.")

        generar_logs(
            tamano_imagen=tamano_imagen,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            acc_train=acc_entrenamiento,
            acc_val=acc_validacion,
            dropout=dropout,
            log_archivo=logs_ruta
        )

        if args.apagar:
            os.system('shutdown /s /f /t 0')

    except KeyboardInterrupt:
        print("\nFinalizado")