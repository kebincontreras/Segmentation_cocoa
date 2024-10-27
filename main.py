import torch

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import fcn_resnet50
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split

"""# Visualización de imágenes y máscaras del dataset"""

root = './data'
dataset = OxfordIIITPet(root, download=True, target_types="segmentation", transforms=None)

image, mask = dataset[0]
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Imagen')
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title('Mascara Real')
plt.show()

"""# Función de normalización para imágenes y máscaras de segmentación"""

def normalize(input_image, input_mask):
    input_image = TF.to_tensor(input_image)
    input_mask = torch.as_tensor(np.array(input_mask), dtype=torch.long) - 1

    return input_image, input_mask

"""# Carga y preprocesamiento de imágenes y máscaras de segmentación"""

def load_image(datapoint):
    input_image, input_mask = datapoint
    input_image = TF.resize(input_image, (128, 128))
    input_mask = TF.resize(input_mask, (128, 128), interpolation=Image.NEAREST)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

transformed_dataset = [load_image(img_mask) for img_mask in dataset]

"""# División del dataset y creación de DataLoaders para entrenamiento y validación
80% entrenamiento

20% validación
"""

train_size = int(0.8 * len(transformed_dataset))
val_size = len(transformed_dataset) - train_size
train_dataset, val_dataset = random_split(transformed_dataset, [train_size, val_size])

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""# Definición de transformaciones y función de aumento de datos para imágenes y máscaras"""

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    ToTensorV2()
])

def augment(image, mask):
    augmented = transform(image=np.array(image), mask=np.array(mask))
    return augmented['image'], augmented['mask']

"""# Función de agrupamiento y creación de DataLoaders"""

def collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack([img for img in images])
    masks = torch.stack([msk for msk in masks])
    return images, masks

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

"""# Función de visualización de imágenes y máscaras, y ejemplo de visualización del dataset"""

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Imagen', 'Mascaara real']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i].cpu().numpy().transpose(1, 2, 0) if i == 0 else display_list[i].cpu().numpy())
        plt.axis('off')
    plt.show()

for images, masks in train_loader:
    sample_image, sample_mask = images[20], masks[20]
    display([sample_image, sample_mask])
    break

"""# Cargar modelo de segmentación preentrenado"""

model = fcn_resnet50(pretrained=True)
model.eval()

"""# realizar predicciones"""

def predict(model, dataloader, num_images=5):
    predictions = []
    count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            output = model(images)['out']

            predicted_masks = output.argmax(dim=1).cpu()
            predictions.append(predicted_masks)

            count += images.size(0)
            if count >= num_images:
                break
    return torch.cat(predictions)[:num_images]
predicted_masks = predict(model, val_loader, num_images=5)

"""# visualizar imágenes de entrada, máscaras verdaderas y máscaras predichas"""

def display_predictions(images, true_masks, predicted_masks, num_images=5):
    plt.figure(figsize=(15, 15))

    for i in range(num_images):
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.title('Imagen')
        plt.imshow(images[i].cpu().numpy().transpose(1, 2, 0))

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.title('Mascara real')
        plt.imshow(true_masks[i].cpu().numpy(), cmap='gray')

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.title('Mascara predicha')
        plt.imshow(predicted_masks[i], cmap='gray')
    plt.show()

for images, masks in val_loader:
    display_predictions(images, masks, predicted_masks)
    break


model.aux_classifier = None  
modelo_guardado = "modelo_segmentacion.pth"
torch.save(model.state_dict(), modelo_guardado)
print(f"Modelo guardado en {os.path.abspath(modelo_guardado)}")


