# import numpy as np
# from utils import *
# #%%
# class Phial():
#     def __init__(self,container_region):
#         self.container_region = container_region
        
#     def edge_color_detection(self,container_region):
#         cnt = edge_detection(container_region,0,10)
#         y_lims = [cnt[i][1][0][1] for i in range(0,len(cnt))]
#         eps = 5
#         diff = np.diff(y_lims)
#         for i in range(0,len(diff)):
#             if np.abs(diff[i]) < eps:
#                 y_lims.remove(y_lims[i])
          
#         if y_lims:
#             y_lims.insert(0,0)
#             y_lims.insert(-1,np.shape(container_region)[0])
        
#         y_lims = list(np.sort(y_lims))

#         return cnt, y_lims
    
#     def unique_count_app(self,a):
#         if len(a)> 0:
#             colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
#         return list(colors[count.argmax()])
    
#     def extracting_colors(self,y_lims):
#         colors = []
#         length = 55
#         for i in range(0,len(y_lims)-1):
#             area_single_color = self.container_region[y_lims[i]:y_lims[i+1],:]
#             if np.shape(area_single_color)[0] > length:
#                 q = int(np.ceil(np.shape(area_single_color)[0]/length))
#                 colors.extend([self.unique_count_app(area_single_color)]*q)
#             else:
#                 colors.append(self.unique_count_app(area_single_color))
#         return colors
    
    #%%
import numpy as np
from utils import edge_detection

class Phial:
    def __init__(self, container_region):
        """Initialize the phial object with the container region (image region)."""
        self.container_region = container_region

    def edge_color_detection(self, container_region):
        """Detect edges and extract color region limits (y-limits) from the container."""
        # Detect edges using the provided utility function
        cnt = edge_detection(self.container_region, 0, 10)

        # Extract y-coordinates of the contour points
        y_lims = [cnt[i][1][0][1] for i in range(len(cnt))]

        # Filter out points that are too close to each other
        eps = 5
        y_lims = [y for i, y in enumerate(y_lims) if i == 0 or np.abs(y - y_lims[i - 1]) >= eps]

        # Insert bounds for the full height of the container region
        if y_lims:
            y_lims.insert(0, 0)
            y_lims.append(self.container_region.shape[0])

        # Sort the y-limits to ensure correct order
        y_lims = sorted(y_lims)

        return cnt, y_lims

    def unique_count_app(self, area):
        """Find the most frequent color in the given region."""
        if area.size > 0:  # Check if the array is not empty
            colors, counts = np.unique(area.reshape(-1, area.shape[-1]), axis=0, return_counts=True)
            return colors[counts.argmax()].tolist()  # Return the most frequent color
        return [0, 0, 0]  # Default to black if empty

    def extracting_colors(self, y_lims):
        """Extract the dominant colors in each region defined by y-limits."""
        colors = []
        length_threshold = 55

        # Iterate over the y-limit pairs to extract colors
        for i in range(len(y_lims) - 1):
            area_single_color = self.container_region[y_lims[i]:y_lims[i + 1], :]
            region_height = area_single_color.shape[0]

            if region_height > length_threshold:
                # Calculate how many times the color should be extended for large regions
                q = int(np.ceil(region_height / length_threshold))
                dominant_color = self.unique_count_app(area_single_color)
                colors.extend([dominant_color] * q)
            else:
                # Add the dominant color for smaller regions
                colors.append(self.unique_count_app(area_single_color))

        return colors
