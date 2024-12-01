from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json 


def visualize_keypoints(tensor_data):
    keypoints_array = np.array(tensor_data)
    image_height = 1080
    image_width = 1920
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    keypoints = keypoints_array[0]  

    # Draw keypoints
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    for kp in keypoints:
        x, y, confidence = kp
        if confidence > 0.5:  
            plt.scatter(x, y, c='green', s=50)  

    plt.title('Keypoint Visualization')
    plt.axis('off')  
    plt.show()

model = YOLO("yolo11n-pose.pt")

results = model.track(r"/Users/benjamintang/Desktop/NEOFitness/WhatsApp Video 2024-11-30 at 23.32.03.mp4", show=True, save=True)

def save_data_json(filename, data):
    """
    Save keypoint and bounding box data to a JSON file.

    Parameters:
    - filename: str, the name of the file to save the data.
    - data: list of dicts, where each dict contains 'image_path', 'boxes', and 'keypoints'.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")

# Example usage
data_to_save = [
    {
        "image_path": "image1.jpg",
        "boxes": [[100, 150, 50, 80], [200, 240, 60, 90]],  # Example bounding boxes
        "keypoints": [[120, 160], [130, 170], [140, 180]]  # Example keypoints
    },
    {
        "image_path": "image2.jpg",
        "boxes": [[300, 350, 70, 100]],
        "keypoints": [[320, 360], [330, 370]]
    }
]

for result in results:
    if hasattr(result, 'keypoints'):
        keypoints = result.keypoints  
        visualize_keypoints(keypoints.data)


