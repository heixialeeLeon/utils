import torch
import numpy as np
import cv2
from skimage import util

DIRECTION_HORIZONTAL = 0
DIRECTION_VERTICAL =1

def show_tensor(*tensors, dim=4,direction=0, timeout=1000):
    is_first_image = True
    show_img = None
    for data in tensors:
        if dim == 4:
            data = torch.squeeze(data).data.cpu().numpy().transpose(1, 2, 0)
        elif dim ==3:
            data = data.cpu().numpy().transpose(1,2,0)
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


