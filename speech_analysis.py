from typing import Dict, Optional, List
import numpy as np
import librosa
from pathlib import Path
import logging

class SpeechAnalysis:
    """
    A class for analyzing speech patterns and comparing speech samples,
    specifically designed for clinical assessment of speech in patients with limited mobility.
    """
    def __init__(
        self,
        reference_path: str,
        compare_path: str,
        sample_rate: int = 16000,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        sample_duration: float = 5.0
    ):
        self.reference_path = Path(reference_path)
        self.compare_path = Path(compare_path)
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.sample_duration = sample_duration
        
        self.weights = weights or {
            'pitch_stability': 0.25,
            'articulation_clarity': 0.25,
            'rhythm_timing': 0.2,
            'volume_consistency': 0.15,
            'phonation_quality': 0.15
        }
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configure logging for the analysis process."""
        logging.basicConfig(level=logging.INFO if self.verbose else logging.WARNING)
        return logging.getLogger("SpeechAnalysis")

    def load_and_preprocess(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio file with speech-specific filtering."""
        try:
            # Load audio file with proper error handling
            try:
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=self.sample_duration)
            except Exception as e:
                self.logger.error(f"Error loading audio file {audio_path}: {str(e)}")
                return np.zeros(int(self.sample_rate * self.sample_duration))

            # Handle empty or invalid audio
            if len(audio) == 0:
                self.logger.warning(f"Empty audio file: {audio_path}")
                return np.zeros(int(self.sample_rate * self.sample_duration))

            # Apply pre-emphasis filter
            pre_emphasis = 0.97
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

            # Normalize audio
            audio = librosa.util.normalize(audio)

            return audio
        except Exception as e:
            self.logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            return np.zeros(int(self.sample_rate * self.sample_duration))

    def pitch_stability(self) -> float:
        """Analyze stability of pitch over time."""
        try:
            ref_audio = self.load_and_preprocess(self.reference_path)
            comp_audio = self.load_and_preprocess(self.compare_path)
            
            # Calculate pitch
            ref_pitch, _ = librosa.piptrack(y=ref_audio, sr=self.sample_rate)
            comp_pitch, _ = librosa.piptrack(y=comp_audio, sr=self.sample_rate)
            
            # Ensure same length for comparison
            min_length = min(ref_pitch.shape[1], comp_pitch.shape[1])
            ref_pitch = ref_pitch[:, :min_length]
            comp_pitch = comp_pitch[:, :min_length]
            
            # Compare pitch stability
            pitch_score = np.corrcoef(ref_pitch.flatten(), comp_pitch.flatten())[0,1]
            
            # Handle NaN and ensure value is in [0, 1] range
            if np.isnan(pitch_score):
                return 0.0
            return max(0.0, min(1.0, float(pitch_score)))
            
        except Exception as e:
            self.logger.error(f"Error in pitch stability analysis: {str(e)}")
            return 0.0

    def articulation_clarity(self) -> float:
        """Measure clarity of articulation using spectral contrast."""
        try:
            ref_audio = self.load_and_preprocess(self.reference_path)
            comp_audio = self.load_and_preprocess(self.compare_path)
            
            # Calculate spectral contrast
            contrast_ref = librosa.feature.spectral_contrast(y=ref_audio, sr=self.sample_rate)
            contrast_comp = librosa.feature.spectral_contrast(y=comp_audio, sr=self.sample_rate)
            
            # Ensure same length for comparison
            min_length = min(contrast_ref.shape[1], contrast_comp.shape[1])
            contrast_ref = contrast_ref[:, :min_length]
            contrast_comp = contrast_comp[:, :min_length]
            
            # Compare articulation clarity
            clarity_score = np.corrcoef(contrast_ref.flatten(), contrast_comp.flatten())[0,1]
            return max(0.0, min(1.0, clarity_score))
            
        except Exception as e:
            self.logger.error(f"Error in articulation clarity analysis: {str(e)}")
            return 0.0

    def rhythm_timing(self) -> float:
        """Analyze speech rhythm and timing patterns."""
        try:
            ref_audio = self.load_and_preprocess(self.reference_path)
            comp_audio = self.load_and_preprocess(self.compare_path)
            
            # Extract onset strength envelopes
            onset_env_ref = librosa.onset.onset_strength(y=ref_audio, sr=self.sample_rate)
            onset_env_comp = librosa.onset.onset_strength(y=comp_audio, sr=self.sample_rate)
            
            # Ensure same length for comparison
            min_length = min(len(onset_env_ref), len(onset_env_comp))
            onset_env_ref = onset_env_ref[:min_length]
            onset_env_comp = onset_env_comp[:min_length]
            
            # Calculate rhythm similarity
            rhythm_score = np.corrcoef(onset_env_ref, onset_env_comp)[0,1]
            return max(0.0, min(1.0, rhythm_score))
            
        except Exception as e:
            self.logger.error(f"Error in rhythm timing analysis: {str(e)}")
            return 0.0

    def volume_consistency(self) -> float:
        """Analyze consistency in volume/amplitude."""
        try:
            ref_audio = self.load_and_preprocess(self.reference_path)
            comp_audio = self.load_and_preprocess(self.compare_path)
            
            # Calculate RMS energy
            rms_ref = librosa.feature.rms(y=ref_audio)
            rms_comp = librosa.feature.rms(y=comp_audio)
            
            # Ensure same length for comparison
            min_length = min(rms_ref.shape[1], rms_comp.shape[1])
            rms_ref = rms_ref[:, :min_length]
            rms_comp = rms_comp[:, :min_length]
            
            # Compare volume consistency
            volume_score = np.corrcoef(rms_ref.flatten(), rms_comp.flatten())[0,1]
            
            # Handle NaN and ensure value is in [0, 1] range
            if np.isnan(volume_score):
                return 0.0
            return max(0.0, min(1.0, float(volume_score)))
            
        except Exception as e:
            self.logger.error(f"Error in volume consistency analysis: {str(e)}")
            return 0.0

    def phonation_quality(self) -> float:
        """Analyze voice quality metrics."""
        try:
            ref_audio = self.load_and_preprocess(self.reference_path)
            comp_audio = self.load_and_preprocess(self.compare_path)
            
            # Calculate MFCCs
            mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=self.sample_rate)
            mfcc_comp = librosa.feature.mfcc(y=comp_audio, sr=self.sample_rate)
            
            # Ensure same length for comparison
            min_length = min(mfcc_ref.shape[1], mfcc_comp.shape[1])
            mfcc_ref = mfcc_ref[:, :min_length]
            mfcc_comp = mfcc_comp[:, :min_length]
            
            # Compare phonation quality
            phonation_score = np.corrcoef(mfcc_ref.flatten(), mfcc_comp.flatten())[0,1]
            
            # Handle NaN and ensure value is in [0, 1] range
            if np.isnan(phonation_score):
                return 0.0
            return max(0.0, min(1.0, float(phonation_score)))
            
        except Exception as e:
            self.logger.error(f"Error in phonation quality analysis: {str(e)}")
            return 0.0

    def clinical_speech_assessment(self) -> Dict[str, float]:
        """
        Perform a comprehensive clinical assessment of speech patterns.
        Returns a dictionary of assessment metrics.
        """
        try:
            # Calculate individual metrics with fallback values
            pitch_stability = self.pitch_stability()
            articulation_clarity = self.articulation_clarity()
            rhythm_timing = self.rhythm_timing()
            volume_consistency = self.volume_consistency()
            phonation_quality = self.phonation_quality()

            # Replace any NaN or None values with 0.0
            metrics = {
                'pitch_stability': pitch_stability if isinstance(pitch_stability, (int, float)) and not np.isnan(pitch_stability) else 0.0,
                'articulation_clarity': articulation_clarity if isinstance(articulation_clarity, (int, float)) and not np.isnan(articulation_clarity) else 0.0,
                'rhythm_timing': rhythm_timing if isinstance(rhythm_timing, (int, float)) and not np.isnan(rhythm_timing) else 0.0,
                'volume_consistency': volume_consistency if isinstance(volume_consistency, (int, float)) and not np.isnan(volume_consistency) else 0.0,
                'phonation_quality': phonation_quality if isinstance(phonation_quality, (int, float)) and not np.isnan(phonation_quality) else 0.0
            }

            # Calculate weighted overall assessment
            weighted_sum = sum(
                metrics[metric] * weight 
                for metric, weight in self.weights.items()
                if metric in metrics
            )
            
            # Add overall assessment to results
            metrics['overall_assessment'] = max(0.0, min(1.0, weighted_sum))

            # Ensure all values are within [0, 1] range
            for key in metrics:
                metrics[key] = max(0.0, min(1.0, float(metrics[key])))

            return metrics

        except Exception as e:
            self.logger.error(f"Error in clinical speech assessment: {str(e)}")
            # Return default values if analysis fails
            return {
                'pitch_stability': 0.0,
                'articulation_clarity': 0.0,
                'rhythm_timing': 0.0,
                'volume_consistency': 0.0,
                'phonation_quality': 0.0,
                'overall_assessment': 0.0
            }

    def generate_report(self, assessment_results: Dict[str, float]) -> str:
        """Generate a clinical report from the speech analysis results."""
        try:
            report = ["Speech Analysis Clinical Report", "=" * 30 + "\n"]
            
            for metric, score in assessment_results.items():
                if metric != 'overall_assessment':
                    report.append(f"{metric.replace('_', ' ').title()}: {score:.2f}")
                    
                    # Add specific feedback based on scores
                    if score < 0.5:
                        report.append(f"Suggestion: Focus on improving {metric.replace('_', ' ')}")
                    elif score < 0.7:
                        report.append(f"Note: {metric.replace('_', ' ')} shows room for improvement")
            
            report.append(f"\nOverall Assessment Score: {assessment_results['overall_assessment']:.2f}")
            
            # Add overall feedback
            overall_score = assessment_results['overall_assessment']
            if overall_score >= 0.9:
                report.append("\nExcellent performance! Keep maintaining this level.")
            elif overall_score >= 0.7:
                report.append("\nGood performance with some room for improvement.")
            elif overall_score >= 0.5:
                report.append("\nFair performance. Regular practice recommended.")
            else:
                report.append("\nNeeds significant practice. Consider focusing on specific areas.")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return "Error generating speech analysis report"

    def get_improvement_suggestions(self, assessment_results: Dict[str, float]) -> List[str]:
        """Generate specific improvement suggestions based on analysis results."""
        suggestions = []
        
        for metric, score in assessment_results.items():
            if metric == 'overall_assessment':
                continue
                
            if score < 0.5:
                if metric == 'pitch_stability':
                    suggestions.append("Practice maintaining steady pitch during speech")
                elif metric == 'articulation_clarity':
                    suggestions.append("Focus on clear pronunciation of each sound")
                elif metric == 'rhythm_timing':
                    suggestions.append("Work on consistent speech rhythm and timing")
                elif metric == 'volume_consistency':
                    suggestions.append("Practice maintaining steady volume while speaking")
                elif metric == 'phonation_quality':
                    suggestions.append("Focus on smooth and clear voice production")
        
        return suggestions