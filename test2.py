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
    Returns the bounding box (x, y, width, height) of the road.
    """
    hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([0, 0, 50])  # Adjust based on road color (e.g., yellow-brown)
    upper_bound = np.array([180, 50, 200])  # Adjust based on road color (e.g., yellow-brown)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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

# Set the car speed to 0.0001 by providing a small acceleration value
def get_action_for_low_speed():
    return np.array([0.0, 0.01, 0.0], dtype=np.float32)

# Start the simulation loop
try:
    while True:
        action = get_action_for_low_speed()
        observation, reward, terminated, truncated, info = env.step(action)

        # Detect the car's bounding box
        car_bounding_box = detect_car_bounding_box(observation)
        if car_bounding_box is not None:
            x, y, w, h = car_bounding_box
            cv2.rectangle(observation, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for car

        # Detect the road's bounding box
        road_bounding_box = detect_road_bounding_box(observation)
        if road_bounding_box is not None:
            x, y, w, h = road_bounding_box
            cv2.rectangle(observation, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for road

        # Show the frame with bounding boxes drawn
        cv2.imshow("Car Racing - Bounding Boxes", observation)

        if terminated or truncated:
            print("Episode finished!")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    env.close()
    cv2.destroyAllWindows()
