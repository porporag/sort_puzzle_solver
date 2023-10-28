import cv2 as cv
import matplotlib.pyplot as plt


def load_img(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    return img

#%%
def focus_on_phials(img):
    x, y, w, h = 0, 400, 1000, 600
    region_of_interest = img[y:y+h, x:x+w]
    plt.imshow(region_of_interest)
    return region_of_interest
#%%
def edge_detection(cut_img,low_thresh,high_thresh):
    gray = cv.cvtColor(cut_img, cv.COLOR_RGB2GRAY) 
    gauss = cv.GaussianBlur(gray, (11, 11), 0) 
    canny = cv.Canny(gauss,low_thresh,high_thresh)
    cnt, _ = cv.findContours( 
    canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return cnt
