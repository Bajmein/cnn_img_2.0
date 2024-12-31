from torch.utils.data import DataLoader
from torch import nn, optim, device


def entrenar_modelo(modelo: nn.Module, dataloader: DataLoader, dispositivo: device, num_epochs: int, lr: float) -> None:
    criterio: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizador: optim.Adam = optim.Adam(modelo.parameters(), lr=lr)

    modelo.to(dispositivo)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        modelo.train()
        ejecucion_perdida: float = 0.0

        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            optimizador.zero_grad()

            outputs = modelo(imagenes)
            perdidas = criterio(outputs, etiquetas)

            perdidas.backward()
            optimizador.step()

            ejecucion_perdida += perdidas.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {ejecucion_perdida / len(dataloader)}')