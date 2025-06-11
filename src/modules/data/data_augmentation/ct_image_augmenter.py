import albumentations as A
import cv2
import numpy
import random
import torch
import torch.nn.functional as F

def patch_shuffle(img, patch_size=2):
    if isinstance(img, numpy.ndarray):
        img = torch.tensor(img)

    if img.dim() != 3:
        raise ValueError("Expected 3D tensor with shape (H, W, C)")

    H, W, C = img.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch size {patch_size}")

    # Reshape to patches: (H//p, W//p, p, p, C)
    patches = img.view(H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = patches.permute(0, 2, 1, 3, 4)  # (n_patches_h, n_patches_w, patch_size, patch_size, C)

    # Shuffle pixels inside each patch
    for i in range(patches.shape[0],):
        for j in range(patches.shape[1]):
            for c in range(C):
                patch = patches[i, j, :, :, c]
                flat = patch.flatten()
                shuffled = flat[torch.randperm(flat.numel())]
                patches[i, j, :, :, c] = shuffled.view(patch_size, patch_size)

    # Reconstruct the image
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    shuffled_img = patches.view(H, W, C)

    return shuffled_img


class PatchShuffleTransform(A.ImageOnlyTransform):
    def __init__(self, patch_size=16, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.patch_size = patch_size

    def apply(self, img, **params):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img_tensor = torch.tensor(img)
        shuffled_img = patch_shuffle(img_tensor, patch_size=self.patch_size)
        return shuffled_img.numpy()

    def get_transform_init_args_names(self):
        return ("patch_size",)


class CTImageAugmenter:
    """
    A class for applying random 2D data augmentations to CT images and masks.

    This class implements random 2D data augmentations based on the BigAug method proposed by Ling 
    Zhang et al. (2020) in the paper "Generalizing Deep Learning for Medical Image Segmentation 
    to Unseen Domains via Deep Stacked Transformation." BigAug consists of a sequence of nine 
    deep stacked transformations designed to enhance the generalization of 3D medical image 
    segmentation models to unseen domains. The transformations are categorized into three groups:

    - **Image Quality**: Includes noise insertion, blur insertion, and sharpness adjustment to 
      simulate variations in image quality.
    - **Image Appearance**: Involves brightness, contrast, and intensity perturbations to emulate 
      the characteristics of different scanners and imaging protocols.
    - **Spatial Configuration**: Comprises rotation, scaling, and deformation to account for intra-
      and inter-patient differences and variations in imaging quality.

    This class differs from the original BigAug framework by omitting sharpness adjustment and 
    intensity perturbation while incorporating vertical and horizontal flips, as well as 
    translation and shearing. Therefore, the transformations covered by this class include 
    rotation, translation, shearing, flipping, elastic deformation, blurring, brightness/contrast 
    adjustments, and noise insertion. The number of augmentations applied per call is determined 
    by the configuration.

    Attributes:
        number_of_augmentations (int): The number of random augmentations to apply.
        data_augmentations (list): A list of augmentation transformations to randomly choose from.
    """
    def __init__(self, parameters):
        """
        Initializes the RandomDataAugmenter2D with the provided configuration.

        Args:
            parameters (DictConfig): A configuration object (from OmegaConf) containing the parameters for augmentations.
                It includes attributes for rotation, translation, shearing, elastic deformation, blurring, and brightness/contrast settings.
                Expected keys:
                  - number_of_augmentations (int): Number of augmentations to apply.
                  - random_rotation_degrees (float): Maximum rotation in degrees.
                  - random_translation_fraction (float): Fraction of image to translate.
                  - random_elastic_alpha (float): Strength of elastic transformation.
                  - random_elastic_sigma (float): Smoothing factor for elastic transformation.
                  - random_gaussian_blur (Dict): Dictionary with kernel size and sigma for Gaussian blur.
                      - kernel_size (int): Range of the Gaussian kernel size in pixels.
                      - sigma (float): Range of the Gaussian kernel standard deviation.
                  - random_contrast_intensity_factor (float): Contrast adjustment factor.
                  - random_brightness_intensity_factor (float): Brightness adjustment factor.
        """
        self.number_of_augmentations = parameters.number_of_augmentations
        self.number_of_augmentations = 1 #TODO Eliminate this line when the bug is fixed in the data augmentation library.
        """self.data_augmentations = [
            A.Rotate(
                limit=parameters.random_rotation_degrees,
                p=1.0
            ),
            A.Affine(
                translate_percent=(
                    -parameters.random_translation_fraction,
                    parameters.random_translation_fraction
                ),
                p=1.0
            ),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.ElasticTransform(
                alpha=parameters.random_elastic_alpha,
                sigma=parameters.random_elastic_sigma,
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=parameters.random_gaussian_blur.kernel_size,
                sigma_limit=parameters.random_gaussian_blur.sigma,
                p=1.0
            ),
            A.GaussNoise(
                std_range=(
                    0.1,
                    0.2),
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=parameters.random_contrast_intensity_factor,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=parameters.random_brightness_intensity_factor,
                contrast_limit=0,
                p=1.0
            )
        ]"""
        # Choose one blur randomly among GaussianBlur, MedianBlur, or MotionBlur
        blur_transforms = [
            A.GaussianBlur(
                blur_limit=parameters.random_gaussian_blur.kernel_size,
                sigma_limit=parameters.random_gaussian_blur.sigma,
                p=1.0
            ),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0)
        ]
        selected_blur = random.choice(blur_transforms)

        # Additional transformations
        self.data_augmentations = [
            A.ShiftScaleRotate(
                shift_limit=parameters.random_translation_fraction,
                scale_limit=0.1,
                rotate_limit=parameters.random_rotation_degrees,
                p=1.0
            ),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.ElasticTransform(
                alpha=parameters.random_elastic_alpha,
                sigma=parameters.random_elastic_sigma,
                alpha_affine=0.0,
                p=1.0
            ),
            selected_blur,
            A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=parameters.random_brightness_intensity_factor,
                contrast_limit=parameters.random_contrast_intensity_factor,
                p=1.0
            ),
            PatchShuffleTransform(patch_size=2, p=1.0)  # Now properly wrapped
        ]

        self.data_augmentations = [
            A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
        ]

    def __call__(self, image, mask=None):
        """
        Applies random augmentations to the CT image, and optionally to the mask.

        Args:
            image (torch.Tensor): The CT image tensor to augment, in shape (C, H, W).
            mask (torch.Tensor, optional): Optional mask tensor to augment alongside the image, 
                in shape (C, H, W). Defaults to None if no mask is provided.

        Returns:
            torch.Tensor or tuple: If `mask` is not provided, returns the augmented image tensor.
            If `mask` is provided, returns a tuple containing the augmented image and mask tensors.

        Legend:
            H = Height of the image
            W = Width of the image
            C = Number of channels in the image
        """
        data_augmenter = A.Compose([
            *random.sample(
                population=self.data_augmentations,
                k=self.number_of_augmentations
            )
        ])

        if image.ndim == 2:  # Check if the numpy array image shape is (H, W)
            image = image.reshape(1, *image.shape)  # Convert numpy array image from (H, W) to (1, H, W).
        image = numpy.transpose(image, axes=(1, 2, 0))  # Convert numpy array image from (C, H, W) to (H, W, C).
        if mask is None:
            transformed_data = data_augmenter(image=image)
            for data_type in transformed_data:
                transformed_data[data_type] = numpy.transpose(
                    transformed_data[data_type],
                    axes=(2, 0, 1)
                )  # Convert numpy array image from (H, W, C) to (C, H, W).
            return transformed_data['image']
        else:
            if mask.ndim == 2:  # Check if the numpy array mask shape is (H, W)
                mask = mask.reshape(1, *mask.shape)  # Convert numpy array mask from (H, W) to (1, H, W).
            mask = numpy.transpose(mask, axes=(1, 2, 0))  # Convert numpy array mask from (C, H, W) to (H, W, C).
            transformed_data = data_augmenter(
                image=image,
                mask=mask
            )
            for data_type in transformed_data:
                transformed_data[data_type] = numpy.transpose(
                    transformed_data[data_type],
                    axes=(2, 0, 1)
                )  # Convert numpy array image from (H, W, C) to (C, H, W).
            return transformed_data['image'], transformed_data['mask']
