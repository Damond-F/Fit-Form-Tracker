from flask import Flask, request
import csv
import io

app = Flask(__name__)

@app.route('/pose', methods=['POST'])
def receive_pose_data():
    # Get the JSON data from the request
    pose_data = request.get_json()

    # Now you can use pose_data as a dictionary
    # For example, you can print the nose coordinates like this:
    print(pose_data['Nose_X'], pose_data['Nose_Y'])

    return 'OK', 200

if __name__ == '__main__':
    app.run(port=5000)
