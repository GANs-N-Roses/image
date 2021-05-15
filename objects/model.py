import torch
import math
from typing import List

from _utils.singleton_decorator import singleton


@singleton
class ObjectDetector:
    def __init__(self):
        self.__model = torch.hub.load('yolov5-master', 'yolov5m', source='local')

    def predict_image(self, image_path: str, confidence: float = 0.5) -> List[int]:
        """
        Predicts objects in space image
        :param image_path:
        :param confidence: confidence threshold for pretrained model
        :return: list of detected objects ids, sorted by confidence
        """
        self.__model.conf = confidence
        predictions = self.__model(image_path)
        # class_index is the last element in each prediction
        return [prediction.tolist()[-1] for prediction in predictions.pred[0]]

    def predict_images(self, images_paths: List[str], confidence: float = 0.5,
                       elements_per_batch: int = 70) -> List[List[int]]:
        """
        Predicts objects in space image
        :param images_paths:
        :param confidence: confidence threshold for pretrained model
        :param elements_per_batch: number of elements to process per iteration
        :return: list of lists with detected objects ids per photo,
                 sorted by confidence and in the same order
                 as specified in images_paths
        """
        self.__model.conf = confidence

        n_batches = math.ceil(len(images_paths) / elements_per_batch)
        last_index = 0
        final_predictions = []

        for _ in range(n_batches):
            if last_index + elements_per_batch > len(images_paths):
                current_paths = images_paths[last_index:]
            else:
                current_paths = images_paths[last_index:last_index + elements_per_batch]

            predictions = self.__model(current_paths)
            final_predictions += [[prediction.tolist()[-1]
                                   for prediction in image_predictions]
                                  for image_predictions in predictions.pred]
        return final_predictions


if __name__ == '__main__':
    print(ObjectDetector().predict_image('/home/guille806/Imágenes/dog.jpg'))
    print(ObjectDetector().predict_images(['/home/guille806/Imágenes/dog.jpg', '/home/guille806/Imágenes/dog.jpg']))