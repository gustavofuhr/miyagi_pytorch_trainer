from torchvision import transforms
from PIL import Image


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
        "resize_with_padding": resize_letterbox
    }
    resize_fn = resize_fns[resize_mode]

    if augmentation_opt == "noaug":
        return _no_augmentation(resize_size, resize_fn), _no_augmentation(resize_size, resize_fn)
    elif augmentation_opt == "simple":
        return simple_augmentation(resize_size, resize_fn)
    elif augmentation_opt == "random_erase":
        return rand_erase_augmentation(resize_size, resize_fn)
    else:
        raise ValueError(f"Unknown augmentation option: {augmentation_opt}. "
                         f"Available options: 'noaug', 'simple', 'random_erase'.")
        # return rand_augmentation(resize_size, augmentation_opt, resize_fn)