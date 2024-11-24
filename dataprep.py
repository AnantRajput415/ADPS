import gymnasium as gym
import cv2
import numpy as np
import os

# Create the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human")

# Reset the environment to get the initial frame
observation, info = env.reset()

# Make sure you have directories to store images and labels
image_dir = "car_racing_images"
label_file = "car_racing_labels.txt"
os.makedirs(image_dir, exist_ok=True)

# Open a file to store the bounding box labels (x, y, w, h) for car and road
with open(label_file, 'w') as f:
    f.write("image_name, car_x, car_y, car_w, car_h, road_x, road_y, road_w, road_h\n")

def save_frame_with_labels(observation, car_bbox, road_bbox, frame_idx):
    """
    Saves the current frame with bounding boxes for car and road.
    """
    # Save the image
    image_path = os.path.join(image_dir, f"frame_{frame_idx}.png")
    cv2.imwrite(image_path, cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
    
    # Save bounding box information (car and road)
    with open(label_file, 'a') as f:
        car_x, car_y, car_w, car_h = car_bbox if car_bbox else (0, 0, 0, 0)
        road_x, road_y, road_w, road_h = road_bbox if road_bbox else (0, 0, 0, 0)
        f.write(f"frame_{frame_idx}.png, {car_x}, {car_y}, {car_w}, {car_h}, {road_x}, {road_y}, {road_w}, {road_h}\n")

def detect_car_bounding_box(observation):
    """
    Detect the car's bounding box using color thresholding and contour detection.
    Returns the bounding box (x, y, width, height) of the car.
    """
    hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([0, 100, 100])  # Lower bound for red
    upper_bound = np.array([10, 255, 255])  # Upper bound for red

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 3 < w < 8 and 7 < h < 15:  # Ignore large objects
            if best_bbox is None or (7 < w * h < best_bbox[2] * best_bbox[3]):
                best_bbox = (x, y, w, h)
    return best_bbox



def detect_road_bounding_box(observation):
    """
    Detect the road's bounding box using color thresholding and contour detection.
    This method is tailored to detect gray-colored roads.
    Returns the bounding box (x, y, width, height) of the road.
    """
    hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)
    
    # Define lower and upper bounds for gray color
    lower_bound = np.array([0, 0, 50])  # Low saturation and medium brightness
    upper_bound = np.array([180, 50, 200])  # Allow all hues with low saturation
    
    # Apply the mask to get the region of the road
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological transformations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_bbox = None
    max_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:  # Keep the largest contour
            max_area = area
            largest_bbox = (x, y, w, h)
    return largest_bbox

def get_action_based_on_road_center(observation):
    """
    Adjust the steering based on the center of the road.
    """
    # Detect road bounding box
    road_bounding_box = detect_road_bounding_box(observation)
    
    if road_bounding_box is None:
        # No road detected, maintain previous behavior (i.e., steer center)
        steering = 0.0
    else:
        x, y, w, h = road_bounding_box
        road_center = x + w // 2  # Calculate the center of the road bounding box
        image_center = observation.shape[1] // 2  # Calculate the center of the image

        # Calculate the difference between the road center and the image center
        steering_diff = (road_center - image_center) / (observation.shape[1] / 2)

        # Steer left or right based on road center position
        steering = np.clip(steering_diff, -1.0, 1.0)
    
    # Keep low acceleration and no braking for smooth motion
    return np.array([steering, 0.1, 0.0], dtype=np.float32)

# Main loop for capturing frames
frame_idx = 0
try:
    while True:
        # Get the observation (frame) and process it
        action = get_action_based_on_road_center(observation)

        observation, reward, terminated, truncated, info = env.step(action)  # No action, just capture frames
        
        # Detect bounding boxes for the car and road
        car_bbox = detect_car_bounding_box(observation)
        road_bbox = detect_road_bounding_box(observation)
        
        # Save the frame and corresponding bounding boxes
        save_frame_with_labels(observation, car_bbox, road_bbox, frame_idx)
        
        # Show the frame with bounding boxes (optional)
        if car_bbox:
            x, y, w, h = car_bbox
            cv2.rectangle(observation, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for car
        if road_bbox:
            x, y, w, h = road_bbox
            cv2.rectangle(observation, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for road
        
        cv2.imshow("Car Racing - Frame with Bounding Boxes", observation)
        
        # Increment frame index
        frame_idx += 1
        
        # Stop after capturing a certain number of frames or on 'q' key press
        if frame_idx >= 1000 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    env.close()
    cv2.destroyAllWindows()
