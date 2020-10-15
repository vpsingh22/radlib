import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

class Transforms:

    def __init__(
        self, padding=(0, 0), crop=(0, 0), horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0, gaussian_blur_prob=0.0, rotate_degree=0.0,
        cutout_prob=0.0, cutout_dim=(8, 8), mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5), train=True
    ):
        transforms_list = []

        if train:
            if sum(padding) > 0:
                transforms_list += [A.PadIfNeeded(
                    min_height=padding[0], min_width=padding[1], always_apply=True
                )]
            if sum(crop) > 0:
                transforms_list += [A.RandomCrop(crop[0], crop[1], always_apply=True)]
            if horizontal_flip_prob > 0:  # Horizontal Flip
                transforms_list += [A.HorizontalFlip(p=horizontal_flip_prob)]
            if vertical_flip_prob > 0:  # Vertical Flip
                transforms_list += [A.VerticalFlip(p=vertical_flip_prob)]
            if gaussian_blur_prob > 0:  # Patch Gaussian Augmentation
                transforms_list += [A.GaussianBlur(p=gaussian_blur_prob)]
            if rotate_degree > 0:  # Rotate image
                transforms_list += [A.Rotate(limit=rotate_degree)]
            if cutout_prob > 0:  # CutOut
                if isinstance(mean, float):
                    fill_value = mean * 255.0
                else:
                    fill_value = tuple([x * 255.0 for x in mean])
                transforms_list += [A.CoarseDropout(
                    p=cutout_prob, max_holes=1, fill_value=fill_value,
                    max_height=cutout_dim[0], max_width=cutout_dim[1]
                )]
        

        transforms_list += [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensor()
        ]

        self.transform = A.Compose(transforms_list)
    
    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image