from flask import Flask, Response, render_template, request
import cv2
import csv
import json
import requests
import mediapipe as mp
import time

app = Flask(__name__)

#reference pose based on the selection
reference_pose = 0
#image file paths for references
image_files = ['C:/Users/yuche/Documents/pose_annotation/image.jpg', 'C:/Users/yuche/Documents/pose_annotation/image2.jpg']

# Open the video camera no 0
video = cv2.VideoCapture(0)

# Initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create a CSV file to store pose data
csv_pose_file = open('pose_data.csv', 'w', newline='')
csv_writer_pose = csv.writer(csv_pose_file)
csv_writer_pose.writerow(['Timestamp', 'Nose_X', 'Nose_Y', 'Left_Shoulder_X', 'Left_Shoulder_Y', 'Right_Shoulder_X', 'Right_Shoulder_Y', 'Left_Elbow_X', 'Left_Elbow_Y', 'Right_Elbow_X', 'Right_Elbow_Y', 'Left_Wrist_X', 'Left_Wrist_Y', 'Right_Wrist_X', 'Right_Wrist_Y', 'Left_Hip_X', 'Left_Hip_Y', 'Right_Hip_X', 'Right_Hip_Y', 'Left_Knee_X', 'Left_Knee_Y', 'Right_Knee_X', 'Right_Knee_Y', 'Left_Ankle_X', 'Left_Ankle_Y', 'Right_Ankle_X', 'Right_Ankle_Y', 'Left_Heel_X', 'Left_Heel_Y', 'Right_Heel_X', 'Right_Heel_Y', 'Left_Foot_Index_X', 'Left_Foot_Index_Y', 'Right_Foot_Index_X', 'Right_Foot_Index_Y'])

csv_reference_file = open('reference_data.csv', 'w', newline='')
csv_writer_reference = csv.writer(csv_reference_file)
csv_writer_reference.writerow(['Nose_X', 'Nose_Y', 'Left_Shoulder_X', 'Left_Shoulder_Y', 'Right_Shoulder_X', 'Right_Shoulder_Y', 'Left_Elbow_X', 'Left_Elbow_Y', 'Right_Elbow_X', 'Right_Elbow_Y', 'Left_Wrist_X', 'Left_Wrist_Y', 'Right_Wrist_X', 'Right_Wrist_Y', 'Left_Hip_X', 'Left_Hip_Y', 'Right_Hip_X', 'Right_Hip_Y', 'Left_Knee_X', 'Left_Knee_Y', 'Right_Knee_X', 'Right_Knee_Y', 'Left_Ankle_X', 'Left_Ankle_Y', 'Right_Ankle_X', 'Right_Ankle_Y', 'Left_Heel_X', 'Left_Heel_Y', 'Right_Heel_X', 'Right_Heel_Y', 'Left_Foot_Index_X', 'Left_Foot_Index_Y', 'Right_Foot_Index_X', 'Right_Foot_Index_Y'])

image_file = image_files[reference_pose]
# Read image
print(f"Processing reference image: {image_file}")  # Debug print
frame = cv2.imread(image_file)
if frame is None:
    print(f"Could not read image: {image_file}")
    continue

# Convert to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the frame for pose detection
reference_results = pose.process(frame_rgb)

# Get pose landmark data
reference_landmarks = reference_results.pose_landmarks
if reference_landmarks:
    nose_landmark = reference_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_shoulder_landmark = reference_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder_landmark = reference_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Get landmark positions
    nose_x, nose_y = nose_landmark.x, nose_landmark.y
    left_shoulder_x, left_shoulder_y = left_shoulder_landmark.x, left_shoulder_landmark.y
    right_shoulder_x, right_shoulder_y = right_shoulder_landmark.x, right_shoulder_landmark.y

    # Write pose data to CSV
    csv_writer_reference.writerow([nose_x, nose_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y])

csv_reference_file.close()


def gen_frames():
    while True:
        success, frame = video.read()  # read the camera frame
        if not success:
            break
        else:
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame with Mediapipe Pose
            pose_results = pose.process(frame_rgb)
            # Draw the pose annotation on the frame
            mp_drawing.draw_landmarks(frame_rgb, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write pose data to CSV
            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                pose_data = [cv2.getTickCount()]
                for landmark in pose_landmarks:
                    pose_data.extend([landmark.x, landmark.y])
                csv_writer_reference.writerow(pose_data)
                csv_pose_file.flush()

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5001)
