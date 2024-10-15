import cv2
import mediapipe as mp
import pyautogui

x1 = x2 = y1 = y2 = None

def distance(x1, x2, y1, y2):
    return (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5))

# Initialize webcam and MediaPipe Hands module
webCam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
hand_ = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles  # For improved styling

# recognize if the hand is in a fist
def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]

    for tip, dip in zip(finger_tips, finger_dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y: #in computer vision, y-coordinates increase as you go down the screen
            return False
    
    return True



# loop to process each frame
while True:
    _, image = webCam.read()
    image = cv2.flip(image, 1)  # Flip to make it mirror-like
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    # Process detected hands
    if hands:
        for hand in hands:
            # Draw the hand landmarks with custom styling
            drawing_utils.draw_landmarks(
                image, 
                hand, 
                hand_.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=3),  # White circles
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1)  # White lines
            )

            # Extract specific landmarks for index finger tip (id 8) and thumb tip (id 4)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger tip
                    cv2.circle(img=image, center=(x, y), radius=5, color=(255, 255, 255), thickness=1)  
                    x1 = x
                    y1 = y
                if id == 4:  # Thumb tip
                    cv2.circle(img=image, center=(x, y), radius=5, color=(255, 255, 255), thickness=1)  
                    x2 = x
                    y2 = y

            if is_fist(hand):
                pyautogui.press('esc')

        # Calculate the distance between the thumb and index finger
        if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
            dist = distance(x1, x2, y1, y2)

        # Find the midpoint between thumb and index finger
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # Draw a circle centered at the midpoint with radius = distance / 2
        radius = int(dist // 2)
        cv2.circle(image, (mid_x, mid_y), radius, color=(255, 255, 255), thickness=1)

        # Adjust the system volume based on the distance 
        if dist > 100:  
            pyautogui.press('volumeup')
        else:
            pyautogui.press('volumedown')

    cv2.imshow('Hand volume control', image)

    # Exit loop on 'ESC' key
    key = cv2.waitKey(10)
    if key == 27:
        break

webCam.release()
cv2.destroyAllWindows()
