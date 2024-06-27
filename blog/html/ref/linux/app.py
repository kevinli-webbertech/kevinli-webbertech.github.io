import os
import secrets
from flask import Flask, request, redirect, url_for, render_template, make_response, jsonify
import sqlite3
from Crypto.Cipher import DES3
import base64
from datetime import datetime, timedelta
from flask_cors import CORS, cross_origin

app = Flask(__name__)
# CORS(app, resources={r"/validate_cookie/": {"origins": "*"}})
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
# app.config['CORS_HEADERS'] = 'Content-Type'


SECRET_KEY = b'secret key here'

# Ensure the key is 24 bytes long
if len(SECRET_KEY) != 24:
    raise ValueError("SECRET_KEY must be 24 bytes long")

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blogusers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            course_name TEXT NOT NULL,
            semester TEXT NOT NULL,
            expiration_date TEXT NOT NULL,
            cookie_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def encrypt_data(data):
    cipher = DES3.new(SECRET_KEY, DES3.MODE_EAX)
    nonce = cipher.nonce
    encrypted_data = cipher.encrypt(data.encode())
    return base64.b64encode(nonce + encrypted_data).decode()

def decrypt_data(encrypted_data):
    decoded_data = base64.b64decode(encrypted_data)
    nonce = decoded_data[:16]
    cipher = DES3.new(SECRET_KEY, DES3.MODE_EAX, nonce=nonce)
    decrypted_data = cipher.decrypt(decoded_data[16:])
    return decrypted_data.decode()


    
    
# @app.before_request
# def before_request():
#     headers = {'Access-Control-Allow-Origin': '*',
#                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
#                'Access-Control-Allow-Headers': 'Content-Type'}
#     if request.method.lower() == 'options':
#         return jsonify(headers), 200

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

@app.route('/')
def index():
    cookie = request.cookies.get('user_cookie')
    if not cookie:
        return redirect(url_for('user_form'))
    return redirect(url_for('validate_cookie'))

  

@app.route('/user_form', methods=['POST'])
@cross_origin(supports_credentials=True)
def user_form():
    data = request.get_json()
    if data is None:
        return "Invalid JSON", 400

    first_name = data.get('first_name')
    last_name = data.get('last_name')
    course_name = data.get('course_name')
    semester = data.get('semester')

    if not all([first_name, last_name, course_name, semester]):
        return "Missing data", 400

    expiration_date = (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
    cookie_data = f"{first_name}_{last_name}_{course_name}_{semester}_{expiration_date}"
    
    response = jsonify({"message": "Data created"})
    response.set_cookie('user_data', cookie_data, max_age=60*60*24*180)  # 180 days
    return response

@app.route('/validate_cookie/', methods=['get'])
@cross_origin(origins=["http://0.0.0.0:8000"])
def validate_cookie():
    user_data = request.cookies.get('user_data')
    if user_data:
        return f"Cookie Data: {user_data}"
    return "Hello, cross-origin-world!"

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
