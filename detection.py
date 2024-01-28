import toolbox
import os
import time
from datetime import datetime
from abc import ABC, abstractclassmethod
import numpy as np

datetime_filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
file_directory = os.path.dirname(os.path.abspath(__file__))

class face_detector(ABC):
    """An abstract class for face detection models."""
    def __init__(self, detector_config, img_url):
        self.img_url = img_url
        self.detector_config = detector_config

    def _read_img(self):
        toolbox.dir(file_directory, 'tmp').create()
        downloaded_img_name = toolbox.url_img(self.img_url, os.path.join(file_directory, 'tmp', datetime_filename)).download()
        rgb_img, gray_img = toolbox.img().read(os.path.join(file_directory, downloaded_img_name), gray=True)
        toolbox.img().remove(downloaded_img_name)
        return rgb_img, gray_img
    
    @abstractclassmethod
    def detector(self, gray_img: np.ndarray, detector_config: tuple, img: np.ndarray):
        """An abstract method for detecting faces in the grayscale image using the detector configuration."""
        pass
    
    def run(self):
        """A handler method that reads the image, detects the faces, measures the execution time, and generates the output."""
        rgb_img, gray_img = self._read_img()

        t1 = time.perf_counter()
        face_locations, detector_name, detector_config = self.detector(gray_img, self.detector_config, rgb_img)
        t2 = time.perf_counter()

        taken_time = t2 - t1
        faces_count = len(face_locations)
        
        msg = f"Detector: '{detector_name}' with config: {detector_config} has taken [Detector pure time]: {taken_time:.3f} s to detect {faces_count} faces"
        print(msg)

        toolbox.logger().add(f'{msg}')

        return face_locations, faces_count, rgb_img