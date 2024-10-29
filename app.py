from flask import Flask, render_template, request, jsonify, send_file
from utils import AudioHandler, ExerciseManager
import os
import json
import random

app = Flask(__name__)
audio_handler = AudioHandler()
exercise_manager = ExerciseManager()

# Ensure uploads directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load intents from JSON file
def load_intents():
    with open('intents.json', 'r') as file:
        return json.load(file)

intents = load_intents()

@app.route('/')
def home():
    """Render the landing page"""
    return render_template('home.html')

@app.route('/speech_therapy')
def speech_therapy():
    """Render the speech therapy assistant page with chatbot"""
    return render_template('speech_therapy.html')

@app.route('/get_audio/<filename>')
def get_audio(filename):
    return send_file(f"{UPLOAD_FOLDER}/{filename}")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '').lower()
    
    # Check for exercise-related queries
    if any(word in user_message for word in ["exercises", "exercise", "what can you do"]):
        exercise_list = exercise_manager.get_exercise_types()
        response = "I can help you with the following types of exercises. Please choose one:"
        return jsonify({
            'response': response,
            'exercises': exercise_list
        })
    
    # Check for specific exercise type queries
    for exercise_type in exercise_manager.get_exercise_types():
        if exercise_type.lower() in user_message:
            exercises = exercise_manager.get_exercises_for_type(exercise_type)
            return jsonify({
                'response': f"Here are the available {exercise_type} exercises:",
                'exercises': [exercise_type]
            })
    
    # Check other intents
    for intent in intents['intents']:
        if intent['intent'] in user_message:
            response = random.choice(intent['responses'])
            exercises = intent.get('exercises', [])
            return jsonify({
                'response': response,
                'exercises': exercises
            })
            
    return jsonify({
        'response': random.choice(intents['intents'][-1]['responses']),
        'exercises': []
    })

@app.route('/load_exercise/<exercise_type>')
def load_exercise(exercise_type):
    """Load specific exercise content"""
    return render_template(f'{exercise_type}.html')

if __name__ == '__main__':
    app.run(debug=True)