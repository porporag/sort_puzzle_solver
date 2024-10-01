import cv2 as cv
import matplotlib.pyplot as plt

def display_image(img, title="Image", cmap=None):
    """Utility function to display an image using matplotlib."""
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_img(path):
    """Load an image from the given path and convert it to RGB format."""
    img = cv.imread(path)
    if img is None:
        raise ValueError(f"Image not found at path: {path}")
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    display_image(img_rgb, title="Loaded Image")
    return img_rgb

def focus_on_phials(img, x=0, y=400, w=1000, h=600):
    """Crop the image to focus on the region of interest (phials)."""
    region_of_interest = img[y:y+h, x:x+w]
    display_image(region_of_interest, title="Region of Interest")
    return region_of_interest

def edge_detection(img, low_thresh, high_thresh):
    """Perform edge detection using the Canny algorithm on a grayscale image."""
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (11, 11), 0)
    edges = cv.Canny(blurred, low_thresh, high_thresh)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    display_image(edges, title="Edge Detection", cmap="gray")
    return contours
