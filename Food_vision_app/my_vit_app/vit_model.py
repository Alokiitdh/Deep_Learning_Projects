import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision import transforms

def vit_model(num_classes=34, seed=42):
    torch.manual_seed(seed)
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.heads = nn.Linear(model.heads.in_features, num_classes)
    return model, vit_transforms()

def vit_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Define your class names here (same order as used in training)
class_names = [
    'Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich', 'Taco', 'Taquito', 'apple_pie', 
    'burger', 'butter_naan', 'chai', 'chapati', 'cheesecake', 'chicken_curry', 'chole_bhature', 'dal_makhani', 
    'dhokla', 'fried_rice', 'ice_cream', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa',
      'momos', 'omelette', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa', 'sushi'
]
