#capture
import cv2
#hands keypoints
import mediapipe as mp
#track velocity of hand and cooldown
import time
#perform automation
import pyautogui
import numpy as np

#initialize hands for mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#initialize webcam
cap = cv2.VideoCapture(0)

#creating window
window_name = 'Hand Keypoint Detection'
cv2.namedWindow(window_name)

#tras_fps
prev_frame_time = 0
new_frame_time = 0

#track swipe dispcment
buffer_size = 10
swipe_threshold = 126  #minimum displacement to consider a swipe
velocity_threshold = 20  #minimum velocity to consider a swipe
dead_zone = 77       #dead zone to ignore minor movements,jittering basically

# Cooldown period in seconds
#it basically sees last reverse swipe when done
cooldown_period = 0.75

#timestamps for last detected swipes haha
last_left_swipe_time = 0
last_right_swipe_time = 0
last_up_swipe_time = 0
last_down_swipe_time = 0

#buffers for storing keypoints 'positions'
wrist_coords_buffer = []
index_finger_coords_buffer = []
middle_finger_coords_buffer = []

def calculate_velocity(coords_buffer, time_interval):
    if len(coords_buffer) < 2:#counldnt calculate.failed
        return 0
    dx = coords_buffer[-1][0] - coords_buffer[0][0]
    dy = coords_buffer[-1][1] - coords_buffer[0][1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance / time_interval

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    #flip the captured laterally inverse image before
    image = cv2.flip(image, 1)
    #bgr to rgb
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #process and find hands ,put in hands array
    results = hands.process(image_rgb)
    #convert back for rendering
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    #draw hand on image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            """mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
       """
            #track the wrist(id=0), index finger tip(id=8), and middle finger tip(id=12)
            wrist = hand_landmarks.landmark[0]
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]
            
            h, w, c = image.shape
            wrist_coords = (int(wrist.x * w), int(wrist.y * h))
            index_finger_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            middle_finger_coords = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))
            
            wrist_coords_buffer.append(wrist_coords)
            index_finger_coords_buffer.append(index_finger_coords)
            middle_finger_coords_buffer.append(middle_finger_coords)
            
            #a stack
            if len(wrist_coords_buffer) > buffer_size:
                wrist_coords_buffer.pop(0)
                index_finger_coords_buffer.pop(0)
                middle_finger_coords_buffer.pop(0)
            
            if len(wrist_coords_buffer) == buffer_size:
                time_interval = buffer_size / fps
                index_velocity = calculate_velocity(index_finger_coords_buffer, time_interval)
                middle_velocity = calculate_velocity(middle_finger_coords_buffer, time_interval)
                
                #calculate "relative" displacements
                wrist_dx = wrist_coords_buffer[-1][0] - wrist_coords_buffer[0][0]
                wrist_dy = wrist_coords_buffer[-1][1] - wrist_coords_buffer[0][1]
                index_dx = index_finger_coords_buffer[-1][0] - index_finger_coords_buffer[0][0]
                index_dy = index_finger_coords_buffer[-1][1] - index_finger_coords_buffer[0][1]
                middle_dx = middle_finger_coords_buffer[-1][0] - middle_finger_coords_buffer[0][0]
                middle_dy = middle_finger_coords_buffer[-1][1] - middle_finger_coords_buffer[0][1]
                
                current_time = time.time()
                
                #check for swipe directions alongwith cooldown logic
                if abs(index_dx) > swipe_threshold and index_velocity > velocity_threshold and abs(wrist_dy) < dead_zone:
                    if index_dx > 0 and current_time - last_left_swipe_time > cooldown_period:
                        print("Swipe Right")
                        pyautogui.hotkey('ctrl', 'right')  #changing virtual desktop
                        last_right_swipe_time = current_time
                    elif index_dx < 0 and current_time - last_right_swipe_time > cooldown_period:
                        print("Swipe Left")
                        pyautogui.hotkey('ctrl', 'left')  #changing change virtual desktop
                        last_left_swipe_time = current_time
                    wrist_coords_buffer.clear()
                    index_finger_coords_buffer.clear()
                    middle_finger_coords_buffer.clear()
                elif abs(index_dy) > swipe_threshold and index_velocity > velocity_threshold and abs(wrist_dx) < dead_zone:
                    if index_dy > 0 and current_time - last_up_swipe_time > cooldown_period:
                        print("Swipe Down")
                        pyautogui.scroll(-42)  # Scroll down
                        last_down_swipe_time = current_time
                    elif index_dy < 0 and current_time - last_down_swipe_time > cooldown_period:
                        print("Swipe Up")
                        pyautogui.scroll(42)  # Scroll up
                        last_up_swipe_time = current_time
                    wrist_coords_buffer.clear()
                    index_finger_coords_buffer.clear()
                    middle_finger_coords_buffer.clear()

    #calculating FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    #draw fps on frame
    fps = int(fps)
    fps_text = f"FPS: {fps}"
    cv2.putText(image_bgr, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Display the image.
    cv2.imshow(window_name, image_bgr)
    
    # Break the loop if the 'q' key is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

#release webcam,close window
cap.release()
cv2.destroyAllWindows()
