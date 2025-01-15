import matplotlib.pyplot as plt
import torch


def visualizar_mapas(modelo, imagen, capa_idx):
    modelo.eval()

    with torch.no_grad():
        activaciones = imagen

        for idx, capa in enumerate(list(modelo.children())):
            activaciones = capa(activaciones)

            if idx == capa_idx:
                break

    canales = activaciones.size(1)
    filas = columnas = int(canales ** 0.5)

    fig, axs = plt.subplots(filas, columnas, figsize=(10, 10))

    for i, ax in enumerate(axs.flatten()):
        if i < canales:
            ax.imshow(activaciones[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')

        else:
            ax.remove()

    plt.tight_layout()
    plt.show()