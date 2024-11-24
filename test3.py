import gymnasium as gym
import numpy as np
import cv2

# Create the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human")

# Reset the environment to get the initial frame
observation, info = env.reset()

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
        if  w < 50 and h < 50:  # Ignore large objects
            if best_bbox is None or (w * h < best_bbox[2] * best_bbox[3]):
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


# Start the simulation loop
try:
    while True:
        # Get the steering action based on the road center
        action = get_action_based_on_road_center(observation)


        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Detect the car's bounding box
        car_bounding_box = detect_car_bounding_box(observation)
        if car_bounding_box is not None:
            x, y, w, h = car_bounding_box
            cv2.rectangle(observation, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for car

        # Detect the road's bounding box and draw it on the frame
        road_bounding_box = detect_road_bounding_box(observation)
        if road_bounding_box is not None:
            x, y, w, h = road_bounding_box
            # Draw the bounding box on the frame using OpenCV
            cv2.rectangle(observation, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

        # Show the frame with the bounding box drawn on it
        cv2.imshow("Car Racing - Road Bounding Box", observation)

        # Break the loop if the episode is done
        if terminated or truncated:
            print("Episode finished!")
            break

        # Wait for a small time to keep the window responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    env.close()
    cv2.destroyAllWindows()
