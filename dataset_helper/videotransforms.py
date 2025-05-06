import numpy as np
import random
import numbers
import torch
import torchvision.transforms as T

class RandomRotation(object):
    """Rotate the video randomly within a given range of degrees.
    Applies the same random rotation to each frame of the video sequence.

    Args:
        degrees (float or int): Range of degrees to select a random rotation angle from.
            A positive value corresponds to counter-clockwise rotation. The angle will
            be randomly selected between -degrees and +degrees.
    """
    def __init__(self, degrees=30):
        self.degrees = degrees

    def __call__(self, imgs):
        # Select a single random angle for the entire video
        angle = random.uniform(-self.degrees, self.degrees)
        return self.rotate_video(imgs, angle)

    def rotate_video(self, imgs, angle):
        """Helper function to apply rotation frame by frame."""
        t, h, w, c = imgs.shape
        # Use torchvision's RandomRotation with a fixed angle range
        rotation_transform = T.RandomRotation(degrees=(angle, angle))
        rotated_imgs = np.zeros_like(imgs)

        # Iterate through frames, convert to tensor, rotate, convert back
        for i in range(t):
            img = torch.from_numpy(imgs[i]).permute(2, 0, 1)  # Convert HxWxC to CxHxW
            rotated_img = rotation_transform(img)  # Apply rotation
            rotated_imgs[i] = rotated_img.permute(1, 2, 0).numpy()  # Convert CxHxW back to HxWxC

        return rotated_imgs

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)


class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of a video sequence.
    Applies the same random jitter to each frame of the video sequence.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        # create a single ColorJitter instance to be applied identically to each frame
        self.color_jitter_transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        color_jittered_imgs = np.zeros_like(imgs)

        # Apply the *same* jitter transform instance to each frame
        for i in range(t):
            img = torch.from_numpy(imgs[i]).permute(2, 0, 1)  # Convert HxWxC to CxHxW
            color_jittered_img = self.color_jitter_transform(img)  # Apply color jitter
            color_jittered_imgs[i] = color_jittered_img.permute(1, 2, 0).numpy()  # Convert CxHxW back to HxWxC

        return color_jittered_imgs

    def __repr__(self):
        # Use stored parameters for representation
        return self.__class__.__name__ + '(brightness={0}, contrast={1}, saturation={2}, hue={3})'.format(
            self.brightness, self.contrast, self.saturation, self.hue)


class RandomCrop(object):
    """Crop the given video sequence at a random location.
    Applies the same crop to all frames in the sequence.

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

        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            # If already desired size, return coords for full image
            return 0, 0, h, w

        # Calculate random top-left corner
        top = random.randint(0, h - th) if h != th else 0
        left = random.randint(0, w - tw) if w != tw else 0
        return top, left, th, tw # Return crop parameters

    def __call__(self, imgs):
        # Determine crop parameters once for the whole sequence
        top, left, height, width = self.get_params(imgs, self.size)

        # Apply the crop to all frames simultaneously using slicing
        imgs = imgs[:, top:top+height, left:left+width, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given video sequence at the center.
    Applies the same crop to all frames in the sequence.

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
        t, h, w, c = imgs.shape
        th, tw = self.size # Target height, target width
        # Calculate top-left corner for center crop
        top = int(np.round((h - th) / 2.))
        left = int(np.round((w - tw) / 2.))

        return imgs[:, top:top+th, left:left+tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given video sequence randomly with a given probability.
    Applies the same flip decision (flip or not flip) to all frames.

    Args:
        p (float): probability of the video being flipped. Default value is 0.3.
    """
    def __init__(self, p=0.3):
 
        self.p = p

    def __call__(self, imgs):

        if random.random() < self.p:
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)