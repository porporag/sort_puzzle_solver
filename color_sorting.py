import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from phial import Phial
from utils import load_img, focus_on_phials, edge_detection
from graph import Board
from solver import dfs, bfs
from itertools import count
from collections import defaultdict

# Load and process the image
path = 'test.jpg'
img = load_img(path)
cut_img = focus_on_phials(img)

# Detect edges and find contours
contours = edge_detection(cut_img, 10, 40)
number_of_phials = len(contours)
print('Number of phials:', number_of_phials)

# Draw contours on the image
cv.drawContours(cut_img, contours, -1, (0, 255, 0), 2)
plt.imshow(cut_img)

# Parameters for extracting regions
offset_rect_y = 30
offset_rect_x = 10

# Global lists for storing color data
global_matrix = []
global_n_colors = []

# Loop through contours to process each phial
for i, contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour)
    container_region = cut_img[y+offset_rect_y:y+h-offset_rect_y, x+offset_rect_x:x+w-offset_rect_x]

    # Extract colors and edges from the container region
    colors = Phial(container_region)
    phial_contours, y_lims = colors.edge_color_detection(container_region)
    
    # Draw edges on the container region
    cv.drawContours(container_region, phial_contours, -1, (0, 255, 0), 2)
    
    # Display the result
    plt.figure(i)
    plt.imshow(container_region)
    
    # Extract and store color information
    n_colors = colors.extracting_colors(y_lims)
    global_matrix.append(n_colors)
    global_n_colors.append(len(n_colors))

# Determine the maximum number of colors
max_color_number = np.max(global_n_colors)

# Substitute RGB color lists with a single number (index)
flat_list = [tuple(color) for sublist in global_matrix for color in sublist]
mapping = defaultdict(count().__next__)

# Map flat list to indices and update the global matrix
mapped_flat_list = [mapping[color] for color in flat_list]
for sublist in global_matrix:
    for i, color in enumerate(sublist):
        sublist[i] = mapping[tuple(color)]

# Convert global matrix into a tuple of tuples
global_matrix = tuple(tuple(np.flip(t)) for t in global_matrix)

# Run BFS-DFS solver on the global matrix
history = dfs(global_matrix, max_color_number)
