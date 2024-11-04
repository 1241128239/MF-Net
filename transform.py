"""

Author: Honggu Liu
"""
from torchvision import transforms

mesonet_data_transforms = {
    'train': transforms.Compose([ 
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)  
    ]),
    'val': transforms.Compose([
        # transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
