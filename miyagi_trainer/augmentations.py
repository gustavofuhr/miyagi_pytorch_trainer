from torchvision import transforms
from PIL import Image
import albumentations as A
import numpy as np
import cv2


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def resize_then_center_crop(resize_size):
    """Resize so shorter side == resize_size, then CenterCrop"""
    return transforms.Compose([
        transforms.Resize(resize_size),  # int or (int, int): If int, shorter side matched
        transforms.CenterCrop(resize_size)
    ])

def resize_stretch(resize_size):
    """Resize (H, W) exactly, may stretch image"""
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),  # tuple forces exact size, distorts if AR mismatches
    ])

def resize_letterbox(resize_size, fill_color=(114, 114, 114)):
    """Resize with aspect ratio, pad to square"""

    class LetterboxTransform:
        def __init__(self, size, fill_color=(114, 114, 114)):
            self.size = (size, size) if isinstance(size, int) else size
            self.fill_color = fill_color

        def __call__(self, img):
            w, h = img.size
            tw, th = self.size
            scale = min(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BILINEAR)
            new_img = Image.new("RGB", (tw, th), self.fill_color)
            new_img.paste(img, ((tw - nw) // 2, (th - nh) // 2))
            return new_img

    return transforms.Compose([
        LetterboxTransform(resize_size, fill_color)
    ])

def center_crop_only(crop_size):
    """CenterCrop without any resizing first (will discard borders)."""
    return transforms.Compose([
        transforms.CenterCrop(crop_size)
    ])

# TODO: make a decent support for augmentation from timm.
# from timm.data.transforms_factory import create_transform
#  "rand-m9-n3-mstd0.5", "rand-mstd1-w0",
# def rand_augmentation(resize_size, rand_string_name):
#     return create_transform(resize_size, is_training=True, auto_augment=rand_string_name), \
#                 create_transform(resize_size, is_training=False, auto_augment=rand_string_name)


def rand_erase_augmentation(resize_size, resize_fn=resize_stretch):
    # Train transform with RandomErasing
    train_transform = transforms.Compose([
        *resize_fn(resize_size).transforms,
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        transforms.RandomErasing()
    ])
    # Val transform WITHOUT RandomErasing
    val_transform = transforms.Compose([
        *resize_fn(resize_size).transforms,
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
    ])
    return train_transform, val_transform


def _no_augmentation(resize_size, resize_fn=resize_stretch):
    resize_transform = resize_fn(resize_size)
    return transforms.Compose([
        *resize_transform.transforms,
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
    ])

def horizontal_flip_augmentation(resize_size, resize_fn=resize_stretch):
    resize_transform = resize_fn(resize_size)
    return (
        transforms.Compose([
            *resize_transform.transforms,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        ]),
        _no_augmentation(resize_size, resize_fn)
    )

def simple_augmentation(resize_size, resize_fn=resize_stretch):
    resize_transform = resize_fn(resize_size)
    return (
        transforms.Compose([
            *resize_transform.transforms,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
        ]),
        _no_augmentation(resize_size, resize_fn)
    )

    
def get_augmentations(resize_size, augmentation_opt, resize_mode="resize_exact"):
    resize_fns = {
        "resize_then_center_crop": resize_then_center_crop,
        "resize_exact": resize_stretch,
        "resize_with_padding": resize_letterbox,
        "center_crop_only": center_crop_only
    }
    resize_fn = resize_fns[resize_mode]

    if augmentation_opt == "noaug":
        return _no_augmentation(resize_size, resize_fn), _no_augmentation(resize_size, resize_fn)
    elif augmentation_opt == "simple":
        return simple_augmentation(resize_size, resize_fn)
    elif augmentation_opt == "horizontal_flip":  
        return horizontal_flip_augmentation(resize_size, resize_fn)
    elif augmentation_opt == "random_erase":
        return rand_erase_augmentation(resize_size, resize_fn)
    elif augmentation_opt == "liveness_single":
        return liveness_single_augmentation(resize_size, resize_fn), _no_augmentation(resize_size, resize_fn)
    else:
        raise ValueError(
            f"Unknown augmentation option: {augmentation_opt}. "
            f"Available options: 'noaug', 'simple', 'horizontal_flip', 'random_erase', 'liveness_single'."
        )


class AlbumentationsTransform:
    """Adapter so we can drop an Albumentations pipeline inside torchvision.Compose."""
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)  # RGB HWC uint8
        out = self.aug(image=arr)["image"]  # returns HWC uint8
        return Image.fromarray(out)
    

def liveness_single_augmentation(resize_size, resize_fn=resize_stretch):
    """
    One shared augmentation for live/spoof with:
      - Mild/strong motion blur (random)
      - Tiny geometric jitter
      - Exposure/Gamma
      - CLAHE (light, occasional)
    """
    resize_transform = resize_fn(resize_size)

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        
        # Motion blur: pick mild or strong, applied with p=0.3 overall
        A.OneOf([
            A.MotionBlur(blur_limit=(6, 9), p=1.0),   # mild
            A.MotionBlur(blur_limit=(9, 12), p=1.0),   # strong
        ], p=0.3),

        # Tiny geometric jitter (very conservative to avoid warping live faces)
        A.Affine(
            scale=(1.05, 1.15),          # zoom in +15%
            translate_percent=(0.0, 0.05),  # shift up to 5%
            p=0.2
        ),

        # Exposure/Gamma (pick one small tweak)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0)   # ≈ γ∈[0.70, 1.30]
        ], p=0.5),

        # CLAHE lightly/occasionally
        A.CLAHE(clip_limit=(1, 2), tile_grid_size=(8, 8), p=0.2),
    ])

    train_transform = transforms.Compose([
        *resize_transform.transforms,
        AlbumentationsTransform(aug),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])

    return train_transform