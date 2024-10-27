import gradio as gr
import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
import os

# Ruta del modelo guardado en la misma ubicación que el script
modelo_path = os.path.join(os.getcwd(), "modelo_segmentacion.pth")

# Cargar el modelo sin el clasificador auxiliar
modelo_cargado = fcn_resnet50(pretrained=False)
modelo_cargado.aux_classifier = None  
modelo_cargado.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
modelo_cargado.eval()

# Transformación de la imagen de entrada
def transformar_imagen(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ])
    return transform(image).unsqueeze(0)  # Añade dimensión batch

# Función para realizar la segmentación
def segmentar_imagen(image):
    image_tensor = transformar_imagen(image)
    with torch.no_grad():
        output = modelo_cargado(image_tensor)['out']
        predicted_mask = output.argmax(1).squeeze().cpu().numpy()
    return predicted_mask

# Función para aplicar la máscara binaria y obtener la imagen segmentada en color
def aplicar_mascara_color(image, mask):
    # Convertir la imagen a PIL si es un array de NumPy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convertir la imagen a RGB y redimensionarla
    image = image.convert("RGB")
    image = image.resize((128, 128), Image.LANCZOS)
    image_np = np.array(image)
    
    # Convertir la máscara a binario (0 y 1)
    mask_binaria = (mask > 0).astype(np.uint8)
    mask_3d = np.repeat(mask_binaria[:, :, np.newaxis], 3, axis=2)  # Extender la máscara a 3 canales
    
    # Aplicar la máscara a la imagen original
    resultado = image_np * mask_3d  # Esto conserva los colores originales en las áreas segmentadas y deja el resto negro

    # Crear la imagen en color combinada
    resultado_color = Image.fromarray(resultado.astype(np.uint8), mode="RGB")
    
    return resultado_color

# Función para Gradio que muestra la imagen segmentada en color y la máscara en un eje separado
def procesar_imagen(image):
    segmented_mask = segmentar_imagen(image)
    resultado_color = aplicar_mascara_color(image, segmented_mask)
    
    # Convertir la máscara a una imagen de escala de grises para visualización
    mask_image = Image.fromarray((segmented_mask * 255).astype(np.uint8), mode="L")
    
    return resultado_color, mask_image

# Crear interfaz de Gradio con la imagen segmentada en color y la máscara en un eje separado
iface = gr.Interface(
    fn=procesar_imagen,
    inputs=gr.Image(label="Upload or Take a Photo"),
    outputs=[
        gr.Image(type="pil", label="Segmented Image in Color"),
        gr.Image(type="pil", label="Segmentation Mask"),
    ],
    examples=[
        ["images/1.webp"],
        ["images/2.jpg"]
    ],
    title="Image Segmentation in Color and Mask",
    description="Upload an image or take a photo to segment it and view the result in color along with the segmentation mask."
)

iface.launch()
