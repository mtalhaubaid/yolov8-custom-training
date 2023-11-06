import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import KMeans



image_colors = get_image_colors('bg.png', 8, True)
print(image_colors)  # Display the list of RGB colors
