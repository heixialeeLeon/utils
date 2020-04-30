import cv2
import numpy as np
from numpy import random

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

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, classes=None):
        for t in self.transforms:
            img, boxes, classes = t(img, boxes,classes)
        return img, boxes, classes

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
    show_img("raw",im,boxes)

    ops = RandomHorizontalMirror()
    im_result,boxes,_ = test_ops(im_origin, boxes,0)
    #im_result, boxes, _ = ops(im_origin, boxes, 0)
    show_img("resize",im_result,boxes)
    cv2.waitKey(0)