import gradio as gr
import torch
from vit_model import vit_model, vit_transforms, class_names
from PIL import Image

# Load model
vit, _ = vit_model(num_classes=34)
vit.load_state_dict(torch.load("vision_model_weights.pth", map_location='cpu'))
vit.to('cpu')
vit.eval()

# Inference function
def predict_image(img: Image.Image):
    image = vit_transforms(img).unsqueeze(0).to('cpu')
    with torch.no_grad():
        outputs = vit(image)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    return class_names[pred_idx]

# Gradio UI
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Food Vision 🍉",
    description="Upload an image of an Indian or Western appetizer to predict its class."
)

interface.launch()
