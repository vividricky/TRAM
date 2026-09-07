import cv2
import numpy as np
from PIL import Image

def calculate_contrast(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    contrast = (cumulative_hist[-1] - cumulative_hist[0]) / (cumulative_hist[-1] + cumulative_hist[0])
    return contrast
def calculate_saturation(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    saturation = np.mean(s)
    return saturation
def calculate_colorfulness(image):
    image = Image.fromarray(image)
    image = image.convert('RGB')
    pixels = list(image.getdata())
    r_range = max(pixels, key=lambda x: x[0])[0] - min(pixels, key=lambda x: x[0])[0]
    g_range = max(pixels, key=lambda x: x[1])[1] - min(pixels, key=lambda x: x[1])[1]
    b_range = max(pixels, key=lambda x: x[2])[2] - min(pixels, key=lambda x: x[2])[2]
    colorfulness = (r_range + g_range + b_range) / 3.0
    return colorfulness

def calculate_white_balance(image):
    img_float = image.astype(float)
    b, g, r = cv2.split(img_float)
    avg_b = cv2.mean(b)[0]
    avg_g = cv2.mean(g)[0]
    avg_r = cv2.mean(r)[0]
    white_balance = (avg_b + avg_g + avg_r) / 3
    white_balance = white_balance/255
    return white_balance
