from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from utils import AudioHandler, ExerciseManager
from speech_analysis import SpeechAnalysis
import os
import json
import random
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

# Add configuration to prevent frequent reloading
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Disable file watcher for specific directories
extra_dirs = ['templates/', 'static/']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in os.walk(extra_dir):
        for filename in files:
            filename = os.path.join(dirname, filename)
            if os.path.isfile(filename):
                extra_files.append(filename)

audio_handler = AudioHandler()
exercise_manager = ExerciseManager()

# Ensure directories exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/articulation_audio', exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

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

@app.route('/submit_audio', methods=['POST'])
def submit_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    word = request.form.get('word', 'unknown')
    
    if audio_file:
        filename = secure_filename(f"{word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        return jsonify({'success': True, 'filename': filename}), 200
    
    return jsonify({'error': 'Invalid audio file'}), 400

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    """Handle speech analysis requests"""
    logger.info("Received speech analysis request")
    temp_file = None
    
    try:
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({
                'status': 'error',
                'message': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']
        sound_id = request.form.get('sound_id')
        
        if not sound_id:
            logger.error("No sound_id provided")
            return jsonify({
                'status': 'error',
                'message': 'No sound ID provided'
            }), 400

        # Create temporary file with proper cleanup
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            # Save the audio file
            audio_file.save(temp_path)
            logger.info(f"Saved uploaded audio to temporary file: {temp_path}")
            
            # Get reference audio path based on sound_id
            reference_path = None
            for folder in ['speech_audio', 'sentence_audio', 'tongue_audio', 'articulation_audio']:
                possible_path = os.path.join('static', folder, f'{sound_id}.mp3')
                if os.path.exists(possible_path):
                    reference_path = possible_path
                    break
            
            if not reference_path:
                logger.error(f"Reference audio not found for sound_id: {sound_id}")
                return jsonify({
                    'status': 'error',
                    'message': 'Reference audio not found'
                }), 404
            
            # Initialize speech analysis
            analyzer = SpeechAnalysis(
                reference_path=reference_path,
                compare_path=temp_path,
                verbose=True
            )
            
            # Perform analysis
            results = analyzer.clinical_speech_assessment()
            suggestions = analyzer.get_improvement_suggestions(results)
            
            # Handle NaN values and ensure consistent structure
            sanitized_results = {
                'pitch_stability': float(results.get('pitch_stability', 0) or 0),
                'articulation_clarity': float(results.get('articulation_clarity', 0) or 0),
                'rhythm_timing': float(results.get('rhythm_timing', 0) or 0),
                'volume_consistency': float(results.get('volume_consistency', 0) or 0),
                'phonation_quality': float(results.get('phonation_quality', 0) or 0),
                'overall_assessment': float(results.get('overall_assessment', 0) or 0)
            }
            
            # Generate feedback based on overall score
            overall_score = sanitized_results['overall_assessment']
            if overall_score >= 0.9:
                feedback = "Excellent pronunciation! Keep up the great work!"
            elif overall_score >= 0.7:
                feedback = "Good pronunciation. Minor improvements possible."
            elif overall_score >= 0.5:
                feedback = "Fair pronunciation. Try focusing on clarity and consistency."
            else:
                feedback = "Keep practicing! Focus on matching the reference audio more closely."
            
            return jsonify({
                'status': 'success',
                'results': sanitized_results,
                'feedback': feedback,
                'suggestions': suggestions
            })
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in speech analysis: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Error analyzing speech'
        }), 500

@app.route('/articulation')
def articulation():
    return render_template('articulation.html')

@app.route('/sentence')
def sentence():
    return render_template('sentence.html')

@app.route('/tongue')
def tongue():
    return render_template('tongue.html')

@app.route('/load_exercise/<exercise_type>')
def load_exercise(exercise_type):
    """Load specific exercise content"""
    return render_template(f'{exercise_type}.html')

@app.route('/static/js/<path:filename>')
def serve_static_js(filename):
    return send_from_directory('static/js', filename)

@app.route('/verify_audio')
def verify_audio():
    """Verify all audio files exist and are accessible"""
    audio_files = {
        'speech_audio': [
            's_sun.mp3',
            's_snake.mp3',
            'r_red.mp3',
            'th_think.mp3'
        ],
        'sentence_audio': [
            'sentence1.mp3',
            'sentence2.mp3'
        ],
        'tongue_audio': [
            'peter_piper.mp3',
            'seashells.mp3'
        ],
        'articulation_audio': [
            'p_pat.mp3',
            'b_ball.mp3',
            's_sun.mp3',
            's_snake.mp3',
            'r_red.mp3',
            'th_think.mp3'
        ]
    }
    
    results = []
    for folder, files in audio_files.items():
        for file in files:
            path = os.path.join('static', folder, file)
            exists = os.path.exists(path)
            results.append({
                'folder': folder,
                'file': file,
                'exists': exists,
                'path': path
            })
    
    return render_template('verify_audio.html', results=results)

@app.route('/verify_files')
def verify_files():
    """Verify all required audio files exist"""
    missing_files = []
    for folder, files in audio_files.items():
        for file in files:
            path = os.path.join('static', folder, file)
            if not os.path.exists(path):
                missing_files.append(f"{folder}/{file}")
    
    if missing_files:
        return jsonify({
            'status': 'error',
            'missing_files': missing_files
        })
    
    return jsonify({
        'status': 'success',
        'message': 'All audio files present'
    })

def check_audio_directories():
    """Ensure all required directories and files exist"""
    required_dirs = [
        'static/speech_audio',
        'static/sentence_audio',
        'static/tongue_audio',
        'static/articulation_audio',
        'static/uploads',
        'static/temp'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Checked directory: {directory}")

# Call this function when the app starts
if __name__ == '__main__':
    check_audio_directories()
    app.run(
        debug=True,
        extra_files=extra_files,
        use_reloader=True,
        reloader_type='stat'
    )