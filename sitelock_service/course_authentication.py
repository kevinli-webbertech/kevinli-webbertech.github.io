from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1>Hello,Blog Server running...</h1>"

# Load the students.json file
with open('students.json', 'r') as f:
    students_data = json.load(f)

# Extract emails from the loaded students_data
students = students_data['emails']

# API to check if the email exists in students.json
@app.route('/check_user', methods=['POST'])
def check_user():
    try:
        # Get the email data from request
        data = request.json
        email = data.get('email')

        # Check if email is in the students list
        if email in students:
            return jsonify({"message": "Valid user", "status": "success"})
        else:
            return jsonify({"message": "Invalid user", "status": "failure"}), 404
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e), "status": "failure"}), 500

if __name__ == '__main__':
    app.run(debug=True)