from flask import Flask, render_template      

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("app.html")

@app.route('/process_frames', methods=['POST'])
def process_frames():
    
    return 'Frames processed and data saved successfully'
    
if __name__ == "__main__":
    app.run(debug=True)
