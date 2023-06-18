from flask import Flask, request
import csv
import io

from dotenv import load_dotenv
import os

import openai

load_dotenv()

key = os.getenv("OPENAI")



openai.api_key = key
model_id = 'gpt-4'

def normalize(pose_data):
    # Translate the pose so that the nose is at the origin
    align_point_x = pose_data['Nose_X']
    align_point_y = pose_data['Nose_Y']

    # Calculate the reference length (distance between the shoulders)
    shoulder_dx = pose_data['Right_shoulder_X'] - pose_data['Left_shoulder_X']
    shoulder_dy = pose_data['Right_shoulder_Y'] - pose_data['Left_shoulder_Y']
    reference_length = math.sqrt(shoulder_dx ** 2 + shoulder_dy ** 2)

    # Calculate the angle to rotate the pose
    reference_angle = math.atan2(shoulder_dy, shoulder_dx)

    # Create a new dictionary to store the normalized pose data
    normalized_pose_data = {}

    # Iterate over the keys and values in the pose data
    for key, value in pose_data.items():
        # If the key ends with '_X' or '_Y', it's a coordinate
        if key.endswith('_X') or key.endswith('_Y'):
            # Subtract the coordinates of the align point
            x = pose_data[key[:-2] + '_X'] - align_point_x
            y = pose_data[key[:-2] + '_Y'] - align_point_y

            # Rotate and scale the coordinates
            if key.endswith('_X'):
                normalized_pose_data[key] = (x * math.cos(reference_angle) + y * math.sin(reference_angle)) / reference_length
            else:
                normalized_pose_data[key] = (-x * math.sin(reference_angle) + y * math.cos(reference_angle)) / reference_length
        else:
            # If it's not a coordinate, just copy the value
            normalized_pose_data[key] = value

    return normalized_pose_data

def feedback(user_form, standard_form):
    # data = lsit of 2 dictionaries
    prompt = 'this is the data for bad forms: ' + '\n'

    for key, value in user_form.items():
        prompt = prompt + key + ' is at ' + str(value) + '. '

    prompt += '\n' + 'now this is the data for good forms: ' + '\n' # good form

    for key, value in standard_form.items():
        prompt = prompt + key + ' is at ' + str(value) + '. '

    input = [
        {'role': 'system', 'content': 'You are a fitness assistant. Respond with how the person can fix their exercise, given the input data as coordinates and the coordinates of ideal form'},
       #  {'role': 'user', 'content': prompt}, # need example data and example response
      #   {'role': 'assistant', 'content': ''},
        {'role': 'user', 'content': prompt}
    ]
    response = openai.ChatCompletion.create(
        model = model_id,
        messages = input
    )
    res = response['choices'][0]['message']['content']
    return res

#OFFLINE VERSION OF receive_pose_data
def get_pose_data():
    # Open the CSV file
    standard_form = ""
    user_form = ""
    with open('pose_data.csv', 'r') as file:
        # Create a DictReader
        standard_form = csv.DictReader(file)

    with open('reference_data.csv', 'r') as file:
        # Create a DictReader
        entire_form = csv.DictReader(file)
        user_form = entire_form[len(entire_form)]

    print(feedback(user_form, standard_form))



#THE FOLLOWING IS THE ORIGINAL RECEIVING POSE DATA USING FLASK
# app = Flask(__name__)
# @app.route('/pose', methods=['POST'])

# def receive_pose_data():
#     # Get the JSON data from the request
#     pose_data = request.get_json()
#     user_form = pose_data["user"]
#     standard_form = pose_data["standard"]
#     res = feedback(user_form, standard_form)
#     return 'OK', 200

# if __name__ == '__main__':
#     app.run(port=5000)
