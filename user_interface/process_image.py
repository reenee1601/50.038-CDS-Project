
import cv2 

def process_single_image(input_image):
    # Reduce the size of the image
    src = cv2.resize(input_image, (0,0), fx=0.15, fy=0.15)

    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Intensify the hair countours in preparation for the inpainting algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the original image depending on the mask
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)

    return dst