"""
Module for FastAPI endpoint
"""

import cv2
from fastapi import FastAPI, File, status, HTTPException
import json
import logging
import numpy as np
from numpy.typing import ArrayLike
import torch
import torchvision.transforms as transform
import uuid

from notebooks.utils import UNet


class Config:
    """
    Config
    """

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = UNet(3).float().to(self._device)
        self._model.load_state_dict(torch.load("artifacts/model.pt"))
        self._model.eval()

    @property
    def model(self):
        """
        The trained model.
        """
        return self._model

    @property
    def device(self):
        """
        Pytorch supported device (cuda or cpu).
        """
        return self._device


app = FastAPI()
config = Config()
mytransformsImage = transform.Compose([transform.ToTensor()])
logging.basicConfig(level=logging.INFO)


@app.post("/car-mask")
async def get_car_mask(file: bytes = File(...)):
    """
    Get Car mask
    """
    img, img_name = get_mask(file)

    lower_color_bounds = np.array([0, 0, 100])
    upper_color_bounds = np.array([30, 30, 255])

    mask = cv2.inRange(img, lower_color_bounds, upper_color_bounds)
    masked = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(f"api/outputs/{img_name}.mask.jpg", masked)

    return json.dumps(masked.tolist())


@app.post("/person-mask")
async def get_person_mask(file: bytes = File(...)):
    """
    Get Person mask
    """
    img, img_name = get_mask(file)

    lower_color_bounds = np.array([100, 0, 0])
    upper_color_bounds = np.array([255, 60, 60])

    mask = cv2.inRange(img, lower_color_bounds, upper_color_bounds)
    masked = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(f"api/outputs/{img_name}.mask.jpg", masked)

    return json.dumps(masked.tolist())


def get_mask(file: bytes = File(...)):
    """
    Get Mask
    """
    try:
        img_np = bytes_to_cv2_img(file)
        img_name = str(uuid.uuid1())
        cv2.imwrite(f"api/inputs/{img_name}.jpg", img_np)
    except Exception as exception:
        logging.error("Failed interpreting input image")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Failed reading image information",
        ) from exception

    height, width, _ = img_np.shape
    img_resized = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_AREA)

    image = mytransformsImage(img_resized).unsqueeze(0).to(config.device)
    output = config.model(image)
    output = output.cpu()[0].detach().permute(1, 2, 0).numpy()
    output = (output * 255).astype(int)
    output[output < 0] = 0
    output_resized = cv2.resize(output, (height, width),
                                interpolation=cv2.INTER_AREA)
    blue, _, red = cv2.split(output_resized)
    output_colored = np.copy(output_resized)
    output_colored[:, :, 0] = red
    output_colored[:, :, 2] = blue

    cv2.imwrite(f"api/outputs/{img_name}.jpg", output_colored)

    return output_resized, img_name


def bytes_to_cv2_img(file: bytes) -> ArrayLike:
    """
    Transform uploaded file to numpy array
    """
    image_np = np.frombuffer(file, np.uint8)
    img_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return img_np
