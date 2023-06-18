from flask import Flask, render_template      

import cv2
import csv
import time
import mediapipe as mp
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("app.html")

@app.route('/process_frames', methods=['GET', 'POST'])
def process_frames():

    # Initialize pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    # Create a CSV file to store pose data
    csv_file = open('pose_data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Nose_X', 'Nose_Y', 'Left_Shoulder_X', 'Left_Shoulder_Y', 'Right_Shoulder_X', 'Right_Shoulder_Y', 'Left_Elbow_X', 'Left_Elbow_Y', 'Right_Elbow_X', 'Right_Elbow_Y', 'Left_Wrist_X', 'Left_Wrist_Y', 'Right_Wrist_X', 'Right_Wrist_Y', 'Left_Hip_X', 'Left_Hip_Y', 'Right_Hip_X', 'Right_Hip_Y', 'Left_Knee_X', 'Left_Knee_Y', 'Right_Knee_X', 'Right_Knee_Y', 'Left_Ankle_X', 'Left_Ankle_Y', 'Right_Ankle_X', 'Right_Ankle_Y', 'Left_Heel_X', 'Left_Heel_Y', 'Right_Heel_X', 'Right_Heel_Y', 'Left_Foot_Index_X', 'Left_Foot_Index_Y', 'Right_Foot_Index_X', 'Right_Foot_Index_Y'])


    def close_webcam():
        # global done
        # done = True
        cap.release()

    # cv2.namedWindow('Output')
    # Set your rate limit (in requests per minute)
    rate_limit = 20  # as per your requirement

    # Calculate the delay (in seconds) between each request
    delay = 60 / rate_limit

    last_sent_time = time.time()

    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        try:
            print(ret)
            # Resize the frame for portrait video
            # frame = cv2.resize(frame, (350, 600))
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame for pose detection
            pose_results = pose.process(frame_rgb)
            # Draw skeleton on the frame
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Display the frame
            cv2.imshow('frame', frame)
            # Get pose landmark data
            pose_landmarks = pose_results.pose_landmarks
            if pose_landmarks:
                nose_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_elbow_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                left_wrist_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_hip_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_knee_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                left_ankle_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_heel_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
                right_heel_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
                left_foot_index_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_foot_index_landmark = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
             
                # Get landmark positions
                nose_x, nose_y = nose_landmark.x, nose_landmark.y
                left_shoulder_x, left_shoulder_y = left_shoulder_landmark.x, left_shoulder_landmark.y
                right_shoulder_x, right_shoulder_y = right_shoulder_landmark.x, right_shoulder_landmark.y
                left_elbow_x, left_elbow_y = left_elbow_landmark.x, left_elbow_landmark.y
                right_elbow_x, right_elbow_y = right_elbow_landmark.x, right_elbow_landmark.y
                left_wrist_x, left_wrist_y = left_wrist_landmark.x, left_wrist_landmark.y
                right_wrist_x, right_wrist_y = right_wrist_landmark.x, right_wrist_landmark.y
                left_hip_x, left_hip_y = left_hip_landmark.x, left_hip_landmark.y
                right_hip_x, right_hip_y = right_hip_landmark.x, right_hip_landmark.y
                left_knee_x, left_knee_y = left_knee_landmark.x, left_knee_landmark.y
                right_knee_x, right_knee_y = right_knee_landmark.x, right_knee_landmark.y
                left_ankle_x, left_ankle_y = left_ankle_landmark.x, left_ankle_landmark.y
                right_ankle_x, right_ankle_y = right_ankle_landmark.x, right_ankle_landmark.y
                left_heel_x, left_heel_y = left_heel_landmark.x, left_heel_landmark.y
                right_heel_x, right_heel_y = right_heel_landmark.x, right_heel_landmark.y
                left_foot_index_x, left_foot_index_y = left_foot_index_landmark.x, left_foot_index_landmark.y
                right_foot_index_x, right_foot_index_y = right_foot_index_landmark.x, right_foot_index_landmark.y
             
                # Write pose data to CSV
                csv_writer.writerow([cv2.getTickCount(), nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y, left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_hip_x, left_hip_y, right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y, left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, left_heel_x, left_heel_y, right_heel_x, right_heel_y, left_foot_index_x, left_foot_index_y, right_foot_index_x, right_foot_index_y])

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
        except(e):
            print(e)
            break
    
        # Check if 'Done' button is pressed (keycode 27 is the ESC key)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            close_webcam()
            break
        elif key == ord('q'):
            break

    # cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    return 'Frames processed and data saved successfully'
    
if __name__ == "__main__":
    app.run()
