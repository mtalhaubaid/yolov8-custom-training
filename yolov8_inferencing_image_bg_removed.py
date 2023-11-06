
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import KMeans



from ultralytics import YOLO
from PIL import Image
from rembg import remove 
import cv2
model = YOLO("yolov8n.pt")



# color finding code here


def get_image_colors(image_path, number_of_colors=8, show_chart=True):
    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def get_colors(image, number_of_colors):
        modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

        clf = KMeans(n_clusters=number_of_colors, n_init=10)
        labels = clf.fit_predict(modified_image)

        counts = Counter(labels)
        counts = dict(sorted(counts.items()))

        center_colors = clf.cluster_centers_
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]

        if show_chart:
            plt.figure(figsize=(8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
            plt.show()  # Display the pie chart

        return rgb_colors

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image from '{}'".format(image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colors = get_colors(image, number_of_colors)
    return colors


# end here





input_image = Image.open(r"grey.jpg")
results = model.predict(source=input_image,conf=0.5, show=True,save=False) # source already setup

Object_boxes = results[0].boxes.data.cpu().numpy()
print(Object_boxes)

for i, box in enumerate(Object_boxes):
    x1, y1, x2, y2, conf, cls = box
    cropped = input_image.crop((x1, y1, x2, y2))
    
    cropped_filename = f'cropped_{i}.jpg'
    cropped.save(cropped_filename)

    output = remove(cropped) 
    bg_removed_filename = f'bg_removed{i}.png'
    output.save(bg_removed_filename)

    image_colors = get_image_colors(bg_removed_filename, 4, True)
    print(image_colors)
    
names = model.names
print(names)
for r in results:
    for c in r.boxes.cls:
        print(names[int(c)])


