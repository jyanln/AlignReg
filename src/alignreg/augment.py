import tensorflow.image as tf_image

from torch.utils.data import Dataset
import torchvision.transforms.functional as torch_transforms

# tensorflow
def tf_contrast(c):
    return lambda img: tf_image.adjust_contrast(img, c)

def tf_saturation(c):
    return lambda img: tf_image.adjust_saturation(img, c)

tf_default_augmentations = [tf_contrast(0.1),
                            tf_contrast(0.2),
                            tf_contrast(0.4),
                            tf_contrast(0.8)]

# torch
class AugmentedDataset(Dataset):
    def __init__(self, dataset, transforms, lazy_augmentation=False):
        self.dataset = dataset
        self.lazy = lazy_augmentation
        if not self.lazy:
            self.augmented = list(map(lambda sample: (sample[0], list(map(lambda transform: transform(sample[0]), transforms)), sample[1]), dataset))
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.lazy:
            return self.augmented[idx]
        else:
            images, label = self.dataset.__getitem__(idx)
            return (images, list(map(lambda transform: transform(images), self.transforms)), label)

def torch_rotate(r):
    return lambda img: torch_transforms.rotate(img, r)

def torch_contrast(c):
    return lambda img: torch_transforms.adjust_contrast(img, c)

torch_default_augmentations = [torch_rotate(-15),
                               torch_rotate(-30),
                               torch_rotate(-45),
                               torch_rotate(-60),
                               torch_contrast(0.5),
                               torch_contrast(0.75),
                               torch_contrast(1.5),
                               torch_contrast(2)]
