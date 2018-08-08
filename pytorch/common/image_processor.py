# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


class TorchImageProcessor:
    """Simple data processors"""

    def __init__(self, image_size, is_color, mean, scale,
                 crop_size=0, pad=28, color='BGR',
                 use_cutout=False,
                 use_mirroring=False,
                 use_random_crop=False,
                 use_center_crop=False,
                 use_random_gray=False):
        """Everything that we need to init"""
        # self.mean = mean
        # self.scale = scale
        # self.crop_size = crop_size
        # self.use_mirroring

        self.data_transform = transforms.Compose([
            # transforms.RandomGrayscale(),
            # transforms.RandomResizedCrop(size=200, scale=(0.75, 0.99)),
            # transforms.RandomRotation(degrees=15),
            transforms.ToPILImage(),
            transforms.Resize((112,96)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
            ])

    def process(self, image_path):
        """
        Returns processed data.
        """
        try:
            image = cv2.imread(image_path)
        except:
            image = image_path
            print('---EXCEPT---'*10)

        # if image is None:
        #     print(image_path)


        # print("before")
        image = self.data_transform(image).numpy()
        # print("---"*30, image.shape)

        # TODO: реализуйте процедуры аугментации изображений используя OpenCV и TorchVision
        # на выходе функции ожидается массив numpy с нормированными значениям пикселей

        return image