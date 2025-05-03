import numpy as np
import random
import numbers
import torch
import torchvision.transforms as T

class RandomRotation(object):
    def __init__(self, degrees=30):
        self.degrees = degrees

    def __call__(self, imgs):
        angle = random.uniform(-self.degrees, self.degrees)
        return self.rotate_video(imgs, angle)

    def rotate_video(self, imgs, angle):
        t, h, w, c = imgs.shape
        rotation_transform = T.RandomRotation(degrees=(angle, angle))
        rotated_imgs = np.zeros_like(imgs)

        for i in range(t):
            img = torch.from_numpy(imgs[i]).permute(2, 0, 1)  # Convert to CxHxW
            rotated_img = rotation_transform(img)  # Apply rotation
            rotated_imgs[i] = rotated_img.permute(1, 2, 0).numpy()  # Convert back to HxWxC

        return rotated_imgs

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        color_jittered_imgs = np.zeros_like(imgs)

        for i in range(t):
            img = torch.from_numpy(imgs[i]).permute(2, 0, 1)  # Convert to CxHxW
            color_jittered_img = self.color_jitter(img)  # Apply color jitter
            color_jittered_imgs[i] = color_jittered_img.permute(1, 2, 0).numpy()  # Convert back to HxWxC

        return color_jittered_imgs

    def __repr__(self):
        return self.__class__.__name__ + '(brightness={0}, contrast={1}, saturation={2}, hue={3})'.format(
            self.brightness, self.contrast, self.saturation, self.hue)


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
