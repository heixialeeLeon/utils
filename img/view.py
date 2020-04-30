import torch
import numpy as np
import cv2
from skimage import util
import types

DIRECTION_HORIZONTAL = 0
DIRECTION_VERTICAL =1

def show_single_tensor_boxes(tensor, boxes, percent=True, timeout=1000):

    def percent_to_absolute(im, boxes):
        h,w,c = im.shape
        boxes[:,0] *= w
        boxes[:,1] *= h
        boxes[:,2] *= w
        boxes[:,3] *= h
        boxes = boxes.astype(np.int)
        return boxes
    data = tensor.cpu().numpy().transpose(1, 2, 0)
    #data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if percent:
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        boxes = percent_to_absolute(data,boxes)
    boxes = boxes.astype(np.int)
    for box in boxes:
        cv2.rectangle(data, (box[0],box[1]),(box[2],box[3]),(255,0,0),thickness=2)
    cv2.imshow("view",data)
    cv2.waitKey(timeout)

def show_batch_tensor(batch_tensor, direction=0, timeout=1000):
    if batch_tensor.ndim ==3:
        tensor_list=[batch_tensor]
    else:
        tensor_list = [item for item in batch_tensor[:, ]]
    show_tensor(tensor_list,direction=0, timeout=1000)

def show_tensor(tensors, direction=0, timeout=1000):
    is_first_image = True
    show_img = None
    for data in tensors:
        print(data.shape)
        channel = 0
        if data.ndim == 4:
            data = torch.squeeze(data).data.cpu().numpy().transpose(1, 2, 0)
            channel = data.shape[1]
        elif data.ndim ==3:
            data = data.cpu().numpy().transpose(1,2,0)
            channel = data.shape[0]
        if channel == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if is_first_image:
            show_img = data
            is_first_image = False
        else:
            if direction == DIRECTION_HORIZONTAL:
                show_img = np.hstack((show_img, data))
            else:
                show_img = np.vstack((show_img, data))
    cv2.imshow("view", show_img)
    cv2.waitKey(timeout)

def show_pil_img(pil_img, timeout=1000, is_float =False):
    im = np.array(pil_img)
    if is_float:
        im = util.img_as_ubyte(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow("view", im)
    cv2.waitKey(timeout)


