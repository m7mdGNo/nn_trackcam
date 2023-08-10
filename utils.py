import cv2
import numpy as np

def blur_region(frame, coords,kernel_size=25):
    x1,y1,x2,y2 = coords
    region = frame[y1:y2, x1:x2]
    blurred_region = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
    frame[y1:y2, x1:x2] = blurred_region
    return frame

def noise_region(frame, coords, noise_percentage=0.5):
    x1,y1,x2,y2 = coords
    region = frame[y1:y2, x1:x2]
    
    noise = np.random.normal(scale=noise_percentage, size=region.shape).astype(np.uint8)
    noisy_region = cv2.add(region, noise)
    
    frame[y1:y2, x1:x2] = noisy_region
    return frame

# Example usage
def main():
    input_image = cv2.VideoCapture(0).read()[1] # Replace with your image path
    coords = [100, 300, 150, 350]  # Example coordinates [x1,y1,x2,y2]
    
    blurred_image = blur_region(input_image.copy(), coords)
    
    cv2.imshow("Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
