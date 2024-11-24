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
    This method focuses on detecting small car objects.
    Returns the bounding box (x, y, width, height) of the car.
    """
    # Convert the image to HSV color space for better color segmentation
    hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds for a typical car color (e.g., red or yellow)
    # These values might need to be adjusted based on your car's color
    lower_bound = np.array([0, 100, 100])  # Lower bound for red
    upper_bound = np.array([10, 255, 255])  # Upper bound for red

    # Apply the mask to get the region of the car
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Optionally, apply some morphological transformations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the best bounding box
    best_bbox = None

    for contour in contours:
        # Get the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out large boxes that may correspond to other objects or background
        if w < 50 and h < 50:  # Use a max size threshold to avoid large contours
            if best_bbox is None or (w * h < best_bbox[2] * best_bbox[3]):  # Keep the smallest box
                best_bbox = (x, y, w, h)

    return best_bbox

# Set the car speed to 0.0001 by providing a small acceleration value
def get_action_for_low_speed():
    # Low acceleration and no steering or braking
    return np.array([0.0, 0.01, 0.0], dtype=np.float32)

# Start the simulation loop
try:
    while True:
        # Apply the low speed action
        action = get_action_for_low_speed()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Detect the car's bounding box in the current frame
        car_bounding_box = detect_car_bounding_box(observation)

        if car_bounding_box is not None:
            x, y, w, h = car_bounding_box
            # Draw the bounding box on the frame using OpenCV
            cv2.rectangle(observation, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Show the frame with the bounding box drawn on it
        cv2.imshow("Car Racing - Bounding Box", observation)

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
