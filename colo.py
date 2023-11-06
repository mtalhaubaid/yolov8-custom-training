# from PIL import Image

# # Open the image file
# image = Image.open(r'D:\code\yolov8-custom-training\bg_removed0.png')  # Replace 'your_image.jpg' with your image file

# # Create a dictionary to store pixel counts for each hex code
# pixel_counts = {}

# # Convert the image to RGB mode (removing alpha or other channels)
# image = image.convert('RGB')

# # Get the pixel data
# pixels = image.load()

# # Iterate through each pixel in the image
# for y in range(image.height):
#     for x in range(image.width):
#         # Get the RGB values of the pixel
#         r, g, b = pixels[x, y]

#         # Convert RGB to hex code
#         hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)

#         # Increment the count for this hex code in the dictionary
#         if hex_code in pixel_counts:
#             pixel_counts[hex_code] += 1
#         else:
#             pixel_counts[hex_code] = 1
# cnt=[]
# cd=[]
# for hex_code, count in pixel_counts.items():
#     cnt.append(count)
#     cd.append(hex_code)

# max_cnt = max(cnt)
# id_cnt = cnt.index(max_cnt)
# max_cnt_cd = cd[id_cnt]
# print(f"max_cnt_cd: {max_cnt_cd}, max_cnt: {max_cnt}")

# cnt.pop(id_cnt)
# cd.pop(id_cnt)

# max_cnt = max(cnt)
# id_cnt = cnt.index(max_cnt)
# max_cnt_cd = cd[id_cnt]
# print(f"max_cnt_cd: {max_cnt_cd}, max_cnt: {max_cnt}")


import cv2
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread(r'D:\code\yolov8-custom-training\bg_removed0.png')
# Convert the image to RGB if it's in BGR format (common for OpenCV)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Apply Felzenszwalb's segmentation
segments = felzenszwalb(image, scale=400, min_size=700)
# Visualize the segmentation
plt.imshow(segments, cmap='tab20b')
plt.show()