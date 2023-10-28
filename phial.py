import numpy as np
from utils import *
#%%
class phial():
    def __init__(self,container_region):
        self.container_region = container_region
        
    def edge_color_detection(self,container_region):
        cnt = edge_detection(container_region,0,6)        
        y_lims = [cnt[i][1][0][1] for i in range(0,len(cnt))]
        eps = 5
        diff = np.diff(y_lims)
        for i in range(0,len(diff)):
            if np.abs(diff[i]) < eps:
                y_lims.remove(y_lims[i])
          
        if y_lims:
            y_lims.insert(0,0)
            y_lims.insert(-1,np.shape(container_region)[0])
        
        y_lims = list(np.sort(y_lims))

        return cnt, y_lims
    
    def unique_count_app(self,a):
        if len(a)> 0:
            colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
        return list(colors[count.argmax()])
    
    def extracting_colors(self,y_lims):
        colors = []
        length = 50
        for i in range(0,len(y_lims)-1):
            area_single_color = self.container_region[y_lims[i]:y_lims[i+1],:]
            if np.shape(area_single_color)[0] > length:
                q = int(np.ceil(np.shape(area_single_color)[0]/length))
                colors.extend([self.unique_count_app(area_single_color)]*q)
            else:
                colors.append(self.unique_count_app(area_single_color))
        return colors