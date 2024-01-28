import cv2
import numpy as np
import sys
import os
import dlib
from preprocessing import *
from vars import *


directory_path = os.path.dirname(os.path.abspath(__file__))
unlabeled_path = os.path.join(directory_path, config["ImgConfig"]["UnlabledDirectory"])
save_augmented_imaged(unlabeled_path)