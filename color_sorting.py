import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from phial import phial
from utils import *

#%%
'''Find phials' contours'''
path = 'test.jpg'
img = load_img(path)
cut_img = focus_on_phials(img)
cnt = edge_detection(cut_img,10,40) 

number_of_phials = len(cnt)
print('Number of phials: ', number_of_phials)
cv.drawContours(cut_img, cnt, -1, (0, 255, 0), 2) 
plt.imshow(cut_img) 
#%%
'''Drawing single phial with edges highlighted'''
global_colori = []
i = 0
offset_rect_y = 30
offset_rect_x = 10

global_matrix = []
for c in cnt:
    x, y, w, h = cv.boundingRect(c)
    container_region = cut_img[y+offset_rect_y:y+h-offset_rect_y,x+offset_rect_x :x+w-offset_rect_x]
    colors = phial(container_region)
    contours,y_lims = colors.edge_color_detection(container_region)
    plt.figure(i)
    cv.drawContours(container_region, contours, -1, (0, 255, 0), 2)
    plt.imshow(container_region)
    n_colors = colors.extracting_colors(y_lims)
    global_matrix.append(n_colors)
    i = i +1

#%%
'''Substitute RGB list with a single number as index'''

flat_list = [item for sublist in global_matrix for item in sublist]
from itertools import count
from collections import defaultdict


mapping = defaultdict(count().__next__)
result = []
for element in flat_list:
    result.append(mapping[tuple(element)])

for element in global_matrix:
    if element:
        for index,j in enumerate(element):
            j = tuple(j)
            element[index] = mapping[j]