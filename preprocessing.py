import cv2
import numpy as np
import sys
import os
import dlib
from vars import *



def rotate_image(image, angle):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def flip_image(image):
    return cv2.flip(image, 1)  # 1 for horizontal flip

def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    # Convert image to float32 for adjustment
    image_float = image.astype(np.float32) / 255.0
    # Adjust brightness and contrast
    adjusted_image = cv2.addWeighted(image_float, contrast_factor, image_float, 0, brightness_factor)
    # Clip values to be in the valid range [0, 1]
    adjusted_image = np.clip(adjusted_image, 0, 1)
    # Convert back to uint8
    adjusted_image = (adjusted_image * 255).astype(np.uint8)
    return adjusted_image
def apply_gaussian_blur(image):
    # Apply Gaussian blur with a random kernel size
    kernel_size = int(np.random.uniform(config["PerprocessingConfig"]["KERNAL"]["value1"], config["PerprocessingConfig"]["KERNAL"]["value2"])) * 2 + 1  # Ensure odd kernel size
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image
def apply_color_jittering(image):
    # Randomly adjust gamma and saturation
    gamma = np.random.uniform(config["PerprocessingConfig"]["GAMMA"]["value1"], config["PerprocessingConfig"]["GAMMA"]["value2"])
    saturation = np.random.uniform(config["PerprocessingConfig"]["SATURATION"]["value1"], config["PerprocessingConfig"]["SATURATION"]["value2"])
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Adjust gamma
    hsv_image[:, :, 2] = np.clip(gamma * hsv_image[:, :, 2], 0, 255)
    # Adjust saturation
    hsv_image[:, :, 1] = np.clip(saturation * hsv_image[:, :, 1], 0, 255)
    # Convert back to BGR
    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return jittered_image

def apply_noise(image):
    # Generate random noise
    noise = np.random.normal(loc=0, scale=10, size=image.shape)
    # Add noise to the image
    noisy_image = cv2.add(image, noise.astype(np.uint8))
    # Clip values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

def apply_histogram_equalization(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    # Merge the equalized channel with the original color channels
    equalized_image = cv2.merge([equalized_image, equalized_image, equalized_image])
    return equalized_image



def image_augmentation(image):
    #reading the image
    image = cv2.imread(image)
    # Randomly rotate the image
    angle = np.random.uniform(-15, 15)
    rotated_image = rotate_image(image, angle)
    # Randomly flip the image horizontally
    flipped_image = flip_image(image)
    # Randomly adjust brightness and contrast
    brightness_factor = np.random.uniform(config["PerprocessingConfig"]["BRIGHTNESS"]["value1"], config["PerprocessingConfig"]["BRIGHTNESS"]["value2"])
    contrast_factor = np.random.uniform(config["PerprocessingConfig"]["CONTRAST"]["value1"], config["PerprocessingConfig"]["CONTRAST"]["value2"])
    adjusted_image = adjust_brightness_contrast(image, brightness_factor, contrast_factor)
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image)
    # Apply color jittering
    jittered_image = apply_color_jittering(image)
    # Apply noise injection
    noisy_image = apply_noise(image)
    #apply_histogram_equalization
    histo_image = apply_histogram_equalization(image)
    #flip_Gaussian blur
    flip_Gaussian_blur = apply_gaussian_blur(flipped_image)
    #flip_brightness
    flip_brightness = adjusted_image = adjust_brightness_contrast(flipped_image, brightness_factor, contrast_factor)
    #flip_histo
    flip_histo = apply_histogram_equalization(flipped_image)
    #flip_noise
    flip_noise = apply_noise(flipped_image)
    #flip_jittered
    flip_jittered = apply_color_jittering(flipped_image)
    #brighness_noise
    brightness_noise = adjust_brightness_contrast(noisy_image, brightness_factor, contrast_factor)
    #brighness_histo
    brightness_histo = adjust_brightness_contrast(histo_image, brightness_factor, contrast_factor)
    #brighness_gaussian
    brighness_gaussian = adjust_brightness_contrast(blurred_image, brightness_factor, contrast_factor)
    #gaussian_jittered
    gaussian_jittered = blurred_image = apply_gaussian_blur(jittered_image)
    #gaussian_histo
    gaussian_histo = blurred_image = apply_gaussian_blur(histo_image)
    #histo_noise
    histo_noise = apply_histogram_equalization(noisy_image)
    #jittered_noise
    jittered_noise = jittered_image = apply_color_jittering(noisy_image)
    #jittered_histo
    jittered_histo = jittered_image = apply_color_jittering(jittered_image)

    return rotated_image, flipped_image, adjusted_image, blurred_image, jittered_image, noisy_image , histo_image, flip_Gaussian_blur, flip_brightness, flip_histo, flip_noise, flip_jittered, brightness_noise, brightness_histo, brighness_gaussian, gaussian_jittered, gaussian_histo, histo_noise, jittered_noise, jittered_histo
    


def is_single_person(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Load pre-trained Haarcascades model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    # Return True if only one face (person) is detected
    return len(faces) == 1


def count_faces_hog(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load pre-trained HOG + SVM model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    # Return true if one face is detected
    return len(faces) == 1


def save_augmented_imaged(directory_path):
    # List all files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(directory_path, image_file)
    
        if(not(is_single_person(image_path))):
            print('Please use a valid personal photo')
            continue
    
        image_file_pure = image_file.replace('.jpeg','').replace('.jpg','').replace('.png','')
        extention = image_file[image_file.find('.'):]
        # Perform augmentation on the current image
        preprocessed_images = image_augmentation(image_path)
    
        # Save or use augmented images for training
        for i, augmented_image in enumerate(preprocessed_images):
            output_path = os.path.join(directory_path + '\\Augmented\\', image_file_pure + '_' + str(i+1) + extention)
            cv2.imwrite(output_path, augmented_image)