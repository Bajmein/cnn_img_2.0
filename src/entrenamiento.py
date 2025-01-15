import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from colorama import Fore, Style
from src.perdida_focal import FocalLoss


class EarlyStopping:
    def __init__(self, paciencia=5, delta=0.01) -> None:
        self.paciencia = paciencia
        self.delta = delta
        self.mejor_val_loss = float('inf')
        self.paciencia_actual = 0

    def __call__(self, val_loss) -> bool:
        if val_loss < self.mejor_val_loss - self.delta:
            self.paciencia_actual = 0
            self.mejor_val_loss = val_loss

        else:
            self.paciencia_actual += 1

        return self.paciencia_actual >= self.paciencia


def entrenar_modelo(modelo: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    dispositivo: torch.device, num_epochs: int, lr: float) -> tuple[torch.nn.Module, float, float]:

    criterio = FocalLoss(gamma=2, alpha=0.7, reduction='mean').to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CyclicLR(optimizador, base_lr=1e-6, max_lr=1e-3, step_size_up=2000, mode="triangular")

    modelo.to(dispositivo)

    early_stopping: EarlyStopping = EarlyStopping()

    acc_entrenamiento, acc_validacion = 0.0, 0.0

    lambda_l1: float = 0.0005

    for epoch in range(num_epochs):
        modelo.train()
        ejecucion_perdida_entrenamiento: float = 0.0
        correctos_entrenamiento: int = 0
        total_entrenamiento: int = 0

        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)

            optimizador.zero_grad()

            outputs = modelo(imagenes)
            perdidas = criterio(outputs, etiquetas)

            l1_norm = sum(param.abs().sum() for param in modelo.parameters())
            perdida_total = perdidas + lambda_l1 * l1_norm

            perdida_total.backward()
            optimizador.step()

            ejecucion_perdida_entrenamiento += perdidas.item()
            _, predicciones = torch.max(outputs, 1)
            correctos_entrenamiento += (predicciones == etiquetas).sum().item()
            total_entrenamiento += etiquetas.size(0)

        modelo.eval()
        ejecucion_perdida_validacion: float = 0.0
        correctos_validacion: int = 0
        total_validacion: int = 0

        with torch.no_grad():
            for imagenes, etiquetas in val_loader:
                imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)

                outputs = modelo(imagenes)
                perdidas = criterio(outputs, etiquetas)

                ejecucion_perdida_validacion += perdidas.item()
                _, predicciones = torch.max(outputs, 1)
                correctos_validacion += (predicciones == etiquetas).sum().item()
                total_validacion += etiquetas.size(0)

        acc_entrenamiento: float = correctos_entrenamiento / total_entrenamiento
        acc_validacion: float = correctos_validacion / total_validacion

        scheduler.step()

        print(f"\n{Fore.LIGHTWHITE_EX}Epoch [{epoch + 1}/{num_epochs}]:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  Entrenamiento   -> Acc: {Fore.GREEN}{acc_entrenamiento:.4f}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  Validación      -> Acc: {Fore.GREEN}{acc_validacion:.4f}{Style.RESET_ALL}")

        if early_stopping(ejecucion_perdida_validacion):
            print(f"\nEntrenamiento detenido en la época {epoch + 1}")
            break

    return modelo, acc_entrenamiento, acc_validacion
