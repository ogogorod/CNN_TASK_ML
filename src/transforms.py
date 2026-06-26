"""Train/eval image transforms. Same IMG_SIZE for both models for a fair comparison."""
from torchvision import transforms
import config

# ImageNet stats — used for the transfer model; harmless for the scratch model too.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
