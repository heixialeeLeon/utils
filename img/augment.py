import cv2
import numpy as np
from numpy import random
import types
import numbers
from PIL import Image, ImageOps, ImageEnhance
import math

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def horizontal_flip(im, boxes=None,classes=None):
    flip_boxes = boxes.copy()
    im = cv2.flip(im,1)
    h,w,_= im.shape
    flip_boxes[:,:,0] = 1 - boxes[:,:,0]
    return im, flip_boxes,classes

def veritcal_flip(im, boxes=None,classes=None):
    flip_boxes = boxes.copy()
    im = cv2.flip(im,0)
    h,w,_= im.shape
    flip_boxes[:,:,1] = 1 - boxes[:,:,1]
    return im, flip_boxes,classes

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img, table)

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)

def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    See `Gamma Correction`_ for more details.
    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    # from here
    # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
    table = np.array([((i / 255.0) ** gamma) * 255 * gain
                      for i in np.arange(0, 256)]).astype('uint8')
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)

def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (numpy ndarray): numpy ndarray to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    rows,cols = img.shape[0:2]
    if center is None:
        center = (cols/2, rows/2)
    M = cv2.getRotationMatrix2D(center,angle,1)
    if img.shape[2]==1:
        return cv2.warpAffine(img,M,(cols,rows))[:,:,np.newaxis]
    else:
        return cv2.warpAffine(img,M,(cols,rows))

class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, classes=None):
        for t in self.transforms:
            img, boxes, classes = t(img, boxes,classes)
        return img, boxes, classes

class Compose_Image(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToAbsoluteCoords(object):
    def __call__(self, img, boxes=None, classes=None):
        height, width, channels = img.shape
        boxes[:,:, 0] *= width
        boxes[:,:, 1] *= height
        boxes = boxes.astype(np.int)
        return img, boxes, classes

class ToPercentCoords(object):
    def __call__(self, img, boxes=None, classes=None):
        height, width, channels = img.shape
        boxes = boxes.astype(np.float)
        boxes[:,:,0] /= width
        boxes[:,:,1] /= height
        return img, boxes, classes

class Resize(object):
    def __init__(self, size=(300,300)):
        self.size = size

    def __call__(self, img, boxes, classes=None):
        img_out = cv2.resize(img, self.size)
        return img_out, boxes, classes

class RandomHorizontalMirror(object):
    def __call__(self, img, boxes, classes):
        if random.randint(2):
            return horizontal_flip(img, boxes,classes)
        else:
            return img, boxes, classes

class RandomVerticalMirror(object):
    def __call__(self, img, boxes, classes):
        if random.randint(2):
            return veritcal_flip(img, boxes,classes)
        else:
            return img, boxes, classes

class RandomRotation(object):
    def __init__(self, degrees, same_size=False):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.same_size = same_size

    def __call__(self, img, boxes, classes):
        w = img.shape[1]
        h = img.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = list()
        for bbox in boxes:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        return rot_img, np.array(rot_text_polys, dtype=np.float32),classes


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
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
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            print('Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            print('Hue jitter enabled. Will slow down loading immensely.')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose_Image(transforms)

        return transform

    def __call__(self, img, boxes, classes):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img = transform(img)
        return img, boxes,classes
        return transform(img),boxes,classes

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


def show_img(name,img, boxes):
    for box in boxes[:,]:
        cv2.rectangle(img,(box[0][0],box[0][1]), (box[1][0],box[1][1]),(255,0,0), thickness=2)
    cv2.imshow(name,img)

def test_ops(im, boxes, classes):
    augment = Compose([
        ToPercentCoords(),
        Resize((500,500)),
        RandomHorizontalMirror(),
        RandomVerticalMirror(),
        ColorJitter(brightness=1,contrast=0.5, saturation=0.8,hue=0.3),
        #RandomRotation(10),
        ToAbsoluteCoords()
    ])
    im, boxes,classes = augment(im,boxes,classes)
    return im,boxes,classes

if __name__ == "__main__":
    file =  "../test_imgs/1.jpg"
    im = cv2.imread(file)
    im_origin = im.copy()
    boxes = list()
    boxes1 = [[10,20],[50,80]]
    boxes2 = [[90,100],[150,120]]
    boxes.append(boxes1)
    boxes.append(boxes2)
    boxes = np.array(boxes)
    #show_img("raw",im,boxes)

    ops = RandomRotation(degrees=10)
    im_result,boxes,_ = test_ops(im_origin, boxes,0)
    #im_result, boxes, _ = ops(im_origin, boxes, 0)
    boxes = boxes.astype(np.int)
    show_img("resize",im_result,boxes)
    cv2.waitKey(0)