class SpeechRecorder {
    constructor(options = {}) {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordedBlob = null;
        this.soundId = options.soundId || '';
        
        // Get DOM elements
        this.startButton = document.querySelector(options.startButtonSelector || '.start-recording');
        this.stopButton = document.querySelector(options.stopButtonSelector || '.stop-recording');
        this.analyzeButton = document.querySelector(options.analyzeButtonSelector || '.analyze-speech');
        this.statusDiv = document.querySelector(options.statusSelector || '.recording-status');
        this.playbackDiv = document.querySelector(options.playbackSelector || '.playback');
        this.resultsDiv = document.querySelector(options.resultsSelector || '.analysis-results');
        this.resultsContent = document.querySelector(options.resultsContentSelector || '.results-content');
        
        // Bind event listeners
        this.startButton.addEventListener('click', () => this.startRecording());
        this.stopButton.addEventListener('click', () => this.stopRecording());
        this.analyzeButton.addEventListener('click', () => this.analyzeSpeech());
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = document.createElement('audio');
                audio.src = audioUrl;
                audio.controls = true;
                
                this.playbackDiv.innerHTML = '';
                this.playbackDiv.appendChild(audio);
                this.recordedBlob = audioBlob;
                this.analyzeButton.disabled = false;
            };
            
            this.audioChunks = [];
            this.mediaRecorder.start();
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            this.statusDiv.textContent = 'Recording...';
            
        } catch (err) {
            console.error('Error accessing microphone:', err);
            this.statusDiv.textContent = 'Error: Could not access microphone';
        }
    }

    stopRecording() {
        this.mediaRecorder.stop();
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        this.statusDiv.textContent = 'Recording stopped';
    }

    async analyzeSpeech() {
        if (!this.recordedBlob) {
            this.statusDiv.textContent = 'No recording available for analysis';
            return;
        }

        const formData = new FormData();
        formData.append('audio', this.recordedBlob);
        formData.append('sound_id', this.soundId);

        try {
            const response = await fetch('/analyze_speech', {
                method: 'POST',
                body: formData
            });

            const results = await response.json();
            
            if (results.status === 'success') {
                // Extract metrics from the results.results object
                const metrics = results.results || {};
                
                this.resultsDiv.classList.remove('hidden');
                this.resultsContent.innerHTML = this.formatResults({
                    pitch_stability: metrics.pitch_stability || 0,
                    articulation_clarity: metrics.articulation_clarity || 0,
                    rhythm_timing: metrics.rhythm_timing || 0,
                    volume_consistency: metrics.volume_consistency || 0,
                    phonation_quality: metrics.phonation_quality || 0,
                    overall_assessment: metrics.overall_assessment || 0,
                    feedback: results.feedback || 'No feedback available',
                    suggestions: results.suggestions || []
                });
            } else {
                this.resultsContent.innerHTML = `<p class="text-red-600">Error: ${results.message}</p>`;
            }
        } catch (err) {
            console.error('Error analyzing speech:', err);
            this.resultsContent.innerHTML = '<p class="text-red-600">Error analyzing speech. Please try again.</p>';
        }
    }

    formatResults(results) {
        // Add null checks and default values
        const formatMetric = (value) => ((value || 0) * 100).toFixed(1);
        
        return `
            <div class="bg-gray-100 p-4 rounded">
                <p class="mb-2"><strong>Pitch Stability:</strong> ${formatMetric(results.pitch_stability)}%</p>
                <p class="mb-2"><strong>Articulation Clarity:</strong> ${formatMetric(results.articulation_clarity)}%</p>
                <p class="mb-2"><strong>Rhythm Timing:</strong> ${formatMetric(results.rhythm_timing)}%</p>
                <p class="mb-2"><strong>Volume Consistency:</strong> ${formatMetric(results.volume_consistency)}%</p>
                <p class="mb-2"><strong>Phonation Quality:</strong> ${formatMetric(results.phonation_quality)}%</p>
                <p class="mb-2"><strong>Overall Score:</strong> ${formatMetric(results.overall_assessment)}%</p>
                <p class="mt-4 ${(results.overall_assessment || 0) >= 0.7 ? 'text-green-600' : 'text-yellow-600'}">
                    <strong>Feedback:</strong> ${results.feedback || 'No feedback available'}
                </p>
                ${(results.suggestions || []).length > 0 ? `
                    <div class="mt-4">
                        <strong>Suggestions for Improvement:</strong>
                        <ul class="list-disc list-inside mt-2">
                            ${results.suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }
} 