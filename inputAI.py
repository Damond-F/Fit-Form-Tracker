import cv2
import csv
import mediapipe as mp

# Initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
done = False

# Create a CSV file to store pose data
csv_file = open('pose_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Nose_X', 'Nose_Y', 'Left_Shoulder_X', 'Left_Shoulder_Y', 'Right_Shoulder_X', 'Right_Shoulder_Y'])

def close_webcam():
    global done
    done = True

cv2.namedWindow('Output')
# Set your rate limit (in requests per minute)
rate_limit = 20  # as per your requirement

# Calculate the delay (in seconds) between each request
delay = 60 / rate_limit

last_sent_time = time.time()

while cap.isOpened() and not done:
    # Read frame
    _, frame = cap.read()
    try:
         # Resize the frame for portrait video
         # frame = cv2.resize(frame, (350, 600))
         # Convert to RGB
         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
         # Process the frame for pose detection
         pose_results = pose.process(frame_rgb)
         
         # Draw skeleton on the frame
         mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
         
         # Display the frame
         cv2.imshow('Output', frame)
         
         # Get pose landmark data
         pose_landmarks = pose_results.pose_landmarks
         if pose_landmarks:
             nose_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
             left_shoulder_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
             right_shoulder_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
             
             # Get landmark positions
             nose_x, nose_y = nose_landmark.x, nose_landmark.y
             left_shoulder_x, left_shoulder_y = left_shoulder_landmark.x, left_shoulder_landmark.y
             right_shoulder_x, right_shoulder_y = right_shoulder_landmark.x, right_shoulder_landmark.y
             
             # Write pose data to CSV
             csv_writer.writerow([cv2.getTickCount(), nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y])

         # Check if it's time to send the pose data to the server
         current_time = time.time()
         if current_time - last_sent_time >= delay:
             # Read the pose data from the CSV file
             csv_file.flush()  # Make sure all data is written to the file
             with open('pose_data.csv', 'r') as file:
                 pose_data = file.read()
        
             # Send the pose data to the server
             response = requests.post('http://localhost:5000/pose', data=pose_data)
        
             if response.status_code != 200:
                 print('There was an error sending the pose data:', response.text)
             last_sent_time = current_time    
    except:
         break
    
    # Check if 'Done' button is pressed (keycode 27 is the ESC key)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        close_webcam()
    elif key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
