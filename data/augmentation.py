import numpy as np
import random


def one_hot(seg, ncols = 4):
    ss_oh = np.zeros((seg.size, ncols), dtype=np.float32)
    ss_oh[np.arange(seg.size), seg.ravel()] = 1
    ss_oh = np.reshape(ss_oh, seg.shape + (ncols,))
    return ss_oh.transpose(2, 0, 1)

def DWIAugment(baseline, cond, target, p_reverse=0.5, p_rec=0.1):
    '''
    Args:
        baseline:  n, w, h baseline image
        cond:  4 vector [bvector, bvalue]
        target: n, w, h target DWI

    Returns:

    '''
    if np.random.uniform(0,1) <= p_reverse:
        cond[:, :3] *= -1

    if np.random.uniform(0,1) <= p_rec:
        cond[:, 3] = 0.
        target = baseline
    return baseline, cond, target


class RandomCrop(object):
    def __init__(self, crop_size=128, pre_center=None):
        self.crop_size = crop_size
        self.max_trail = 20
        if pre_center:
            self.center_crop=True
            self.range = pre_center
        else:
            self.center_crop = False

    def __call__(self, images, masks=None):
        assert len(images) >= 1, 'input images as a list'
        if self.center_crop:
            new_list = list()
            xmin, xmax, ymin, ymax = self.range
            for img in images:
                img = img[xmin:xmax, ymin:ymax]
                new_list.append(img)
            images = new_list
        # reference image
        image = images[0]
        x, y, c = image.shape
        images_out = list()
        rand_range_x = x - self.crop_size
        rand_range_y = y - self.crop_size
        if rand_range_x <= 0:
            x_offset = 0
        else:
            x_offset = np.random.randint(rand_range_x)#rand_range_x = 0
        if rand_range_y <= 0:
            y_offset = 0
        else:
            y_offset = np.random.randint(rand_range_y)
        tmp = image[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size]
        i = 0
        while np.sum(tmp) == 0. and i < self.max_trail and rand_range_x > 0 and rand_range_y > 0:
            x_offset = np.random.randint(rand_range_x)
            y_offset = np.random.randint(rand_range_y)
            tmp = image[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size]
            i += 1

        for img in images:
            c = img.shape[-1]
            imgout = np.zeros([self.crop_size, self.crop_size, c])
            tmp = img[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size]
            imgout[:x, :y] = tmp
            images_out.append(imgout)

        if masks is not None:
            maskout = np.zeros([self.crop_size, self.crop_size])
            maskout[:x, :y] = masks[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size]
            return images_out, maskout
        else:
            return images_out, None




'''Geometric augmentation on image+masks'''
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        seg (Image): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, seg)
        img (Image): the cropped image
        seg (Image): the cropped segmentation
    """
    def __init__(self, pre_center=False, range=()):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.pre_crop = pre_center
        if pre_center:
            self.xmin, self.xmax, self.ymin, self.ymax = range

    def __call__(self, image, masks):
        if self.pre_crop:
            image = image[self.xmin:self.xmax, self.ymin:self.ymax]
            masks = masks[self.xmin:self.xmax, self.ymin:self.ymax]
        height, width, _ = image.shape
        mheight, mwidth, _ = masks.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, masks

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_mask = masks

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                mw = w * (mwidth/ width)
                mh = h * (mheight/ height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2 or mh / mw < 0.5 or mh / mw > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)
                mleft = left * (mwidth/ width)
                mtop = top* (mheight/ height)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # convert to integer rect x1,y1,x2,y2
                mrect = np.array([int(mleft), int(mtop), int(mleft + mw), int(mtop + mh)])

                # cut the crop from the image
                current_mask = current_mask[mrect[1]:mrect[3], mrect[0]:mrect[2], :]
                if np.sum(current_mask) == 0:
                    continue
                return current_image, current_mask



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels


