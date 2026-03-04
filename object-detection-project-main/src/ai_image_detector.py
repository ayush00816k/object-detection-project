import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from scipy import fftpack
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


class AIImageDetector:
    """
    Universal AI Image Detector
    Detects AI-generated vs Real images for ANY object type

    Improvements added:
    - Sensitivity option ('low','medium','high') to control artifact thresholds
    - Auto-load local trained model at 'checkpoints/ai_detector.pth' when available
    - Training script available at 'src/train_ai_detector.py' to fine-tune a ResNet18 binary classifier
    """
    
    def __init__(self, model_path=None, method='hybrid', sensitivity='medium'):
        """
        Initialize AI Image Detector
        
        Args:
            model_path: Path to custom trained model (optional)
            method: 'huggingface', 'custom', 'artifact', or 'hybrid'
            sensitivity: 'low'|'medium'|'high' - higher sensitivity favors AI detection (may increase false positives)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.model = None
        self.sensitivity = sensitivity  # new
        
        # Default sensitivity is respected
        # if self.sensitivity == 'medium':
        #     self.sensitivity = 'high'
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # If a local custom checkpoint exists, prefer custom model for better precision
        local_ckpt = os.path.join('checkpoints', 'ai_detector.pth')
        if os.path.exists(local_ckpt) and (method in ['custom','hybrid']):
            try:
                print(f"🔄 Found local AI detector checkpoint at {local_ckpt}, loading custom model...")
                self._load_custom_model(local_ckpt)
                self.method = 'custom'
                return
            except Exception as e:
                print(f"⚠️ Failed to load local checkpoint: {e}")
                # fallback to requested method

        # Load model based on method
        if method == 'huggingface':
            self._load_huggingface_model()
        elif method == 'custom' and model_path:
            self._load_custom_model(model_path)
        elif method == 'hybrid':
            # Try HuggingFace first, fall back to artifact analysis
            try:
                self._load_huggingface_model()
            except:
                print("⚠️ HuggingFace model not available, using artifact analysis")
                self.method = 'artifact'
    
    def _load_huggingface_model(self):
        """Load pre-trained AI detector from Hugging Face"""
        try:
            from transformers import pipeline
            print("🔄 Loading AI detection model from Hugging Face...")
            
            # This model works on ALL image types
            self.classifier = pipeline(
                "image-classification",
                model="umm-maybe/AI-image-detector",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ AI detection model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading HuggingFace model: {e}")
            print("💡 Install transformers: pip install transformers")
            raise
    
    def _load_custom_model(self, model_path):
        """Load your custom trained AI detection model directly using torchvision ResNet-18
        This avoids requiring an external `model_resnet` module and makes checkpoint loading robust.
        """
        try:
            print(f"🔄 Loading custom AI detector from {model_path}...")
            # Build a ResNet-18 and adapt final layer for binary classification
            from torchvision import models
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)

            # Load state dict (handle either state_dict or full model saved)
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and all(k.startswith('fc') or k.startswith('layer') or k.startswith('conv') or k.startswith('bn') for k in state.keys()):
                model.load_state_dict(state)
            else:
                # If it's a full model object or different format, try loading into state_dict
                try:
                    model.load_state_dict(state)
                except Exception:
                    # Last resort: if the saved object is actually a model, assign directly
                    model = state

            self.model = model.to(self.device)
            self.model.eval()
            print("✅ Custom model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading custom model: {e}")
            raise
    
    def analyze_artifacts(self, image_path):
        """
        Analyze image for AI generation artifacts
        
        AI images typically have:
        1. Low sensor noise (cameras always add noise)
        2. Unusual frequency patterns in FFT
        3. Too-perfect pixel distributions
        4. Inconsistent compression artifacts
        5. Unnatural edge consistency
        
        This works for ANY object: laptop, mobile, chair, etc.
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ═══════════════════════════════════════
            # 1️⃣ SENSOR NOISE ANALYSIS
            # ═══════════════════════════════════════
            # Real cameras ALWAYS add sensor noise
            # AI generators create images with very low noise
            
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            noise_diff = gray.astype(float) - denoised.astype(float)
            noise_level = np.std(noise_diff)
            
            # ═══════════════════════════════════════
            # 2️⃣ FREQUENCY DOMAIN ANALYSIS (FFT)
            # ═══════════════════════════════════════
            # Natural images have specific frequency patterns
            # AI images often have unusual high-frequency content
            
            fft = fftpack.fft2(gray)
            fft_shift = fftpack.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Calculate frequency statistics
            fft_mean = np.mean(magnitude)
            fft_std = np.std(magnitude)
            fft_ratio = fft_std / (fft_mean + 1e-10)
            
            # ═══════════════════════════════════════
            # 3️⃣ EDGE CONSISTENCY ANALYSIS
            # ═══════════════════════════════════════
            # AI images sometimes have too-perfect edges
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Edge smoothness
            edge_variance = np.var(edges)
            
            # ═══════════════════════════════════════
            # 4️⃣ COLOR DISTRIBUTION ANALYSIS
            # ═══════════════════════════════════════
            # Natural photos have specific color characteristics
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Color channel variance
            r_var = np.var(img_rgb[:,:,0])
            g_var = np.var(img_rgb[:,:,1])
            b_var = np.var(img_rgb[:,:,2])
            color_variance = (r_var + g_var + b_var) / 3
            
            # ═══════════════════════════════════════
            # 5️⃣ TEXTURE UNIFORMITY
            # ═══════════════════════════════════════
            # AI images can be too uniform in texture
            
            texture_std = np.std(gray)
            
            # Local texture variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # ═══════════════════════════════════════
            # 🎯 SCORING ALGORITHM
            # ═══════════════════════════════════════
            
            ai_score = 0
            explanations = []
            
            # Adjust thresholds by sensitivity to increase detection precision when requested
            sensitivity_map = {
                'low': 0.8,
                'medium': 1.0,
                'high': 1.3
            }
            s_thresh = sensitivity_map.get(self.sensitivity, 1.0)
            
            # Don't inflate scores as aggressively as thresholds
            # This prevents false positives on real high-res images (plants, nature)
            # Use 1.0 for medium/low, and only slight boost (1.1) for high
            s_score_mult = 1.0
            if self.sensitivity == 'high':
                s_score_mult = 1.1
            elif self.sensitivity == 'low':
                s_score_mult = 0.9

            # Rule 1: Very low noise is suspicious
            # Higher sensitivity -> treat higher noise levels as suspicious
            noise_thresh_1 = 2.5 * s_thresh
            noise_thresh_2 = 4.0 * s_thresh
            if noise_level < noise_thresh_1:
                ai_score += 40 * s_score_mult
                explanations.append(f"Very low sensor noise ({noise_level:.2f}) - typical of AI generation")
            elif noise_level < noise_thresh_2:
                ai_score += 25 * s_score_mult
                explanations.append(f"Low sensor noise ({noise_level:.2f}) - slightly suspicious")
            else:
                explanations.append(f"Normal sensor noise ({noise_level:.2f}) - natural camera behavior")

            # Rule 2: Unusual frequency patterns
            fft_low = 0.03 / s_thresh
            fft_high = 0.6 * s_thresh
            if fft_ratio < fft_low or fft_ratio > fft_high:
                ai_score += 25 * s_score_mult
                explanations.append(f"Unusual frequency patterns ({fft_ratio:.3f}) - may be AI-generated")

            # Rule 3: Edge consistency
            # Lower threshold under high sensitivity
            edge_thresh = 0.15 / s_thresh
            if edge_density > edge_thresh:
                ai_score += 15 * s_score_mult
                explanations.append(f"High edge density ({edge_density:.3f}) - too many perfect edges")

            # Rule 4: Color distribution
            color_thresh = 400 / s_thresh
            if color_variance < color_thresh:
                ai_score += 15 * s_score_mult
                explanations.append(f"Low color variance ({color_variance:.1f}) - unnaturally uniform")

            # Rule 5: Texture uniformity
            texture_thresh = 25 * s_thresh
            if texture_std < texture_thresh:
                ai_score += 10 * s_score_mult
                explanations.append(f"Very smooth texture ({texture_std:.1f}) - possibly AI smoothing")

            # Additional check: JPEG quantization table heuristic (if JPEG)
            try:
                quant = self._get_jpeg_qtable(image_path)
                if quant is not None:
                    quant_score = 0
                    # Highly uniform quantization suggests synthetic editing / large-scale recompression
                    q_std = np.std(quant)
                    if q_std < 1.0:
                        quant_score += 10 * s_score_mult
                        explanations.append(f"Very uniform JPEG quantization (std={q_std:.2f}) - suspicious")
                    elif q_std < 3.0:
                        quant_score += 5 * s_score_mult
                        explanations.append(f"Low JPEG quantization variance (std={q_std:.2f}) - possibly suspicious")
                    ai_score += quant_score
            except Exception:
                pass
            
            # Final verdict
            is_ai = ai_score >= 50
            confidence = min(ai_score, 100) if is_ai else min(100 - ai_score, 100)
            
            return {
                'is_ai': is_ai,
                'confidence': float(confidence),
                'method': 'artifact_analysis',
                'metrics': {
                    'noise_level': float(noise_level),
                    'fft_ratio': float(fft_ratio),
                    'edge_density': float(edge_density),
                    'color_variance': float(color_variance),
                    'texture_std': float(texture_std),
                    'ai_score': float(ai_score)
                },
                'explanations': explanations
            }    

        except Exception as e:
            print(f"❌ Error in artifact analysis: {e}")
            return None
        
        except Exception as e:
            print(f"❌ Error in artifact analysis: {e}")
            return None
    
    def predict_huggingface(self, image_path):
        """Predict using Hugging Face pre-trained model"""
        try:
            image = Image.open(image_path).convert('RGB')
            result = self.classifier(image)
            
            # Parse result
            label = result[0]['label'].lower()
            confidence = result[0]['score'] * 100
            
            # Map label strings more robustly and mark cases that are ambiguous
            ai_keywords = ['artificial', 'fake', 'ai', 'generated', 'synthetic', 'computer-generated']
            is_ai = any(word in label for word in ai_keywords)
            
            # If confidence is low but label includes e.g. 'cartoon' or 'painting', treat as suspicious
            if not is_ai and any(k in label for k in ['painting','render','cartoon','illustration']):
                # Increase suspicion for non-photoreal labels
                confidence = max(confidence, 65.0)
                is_ai = True if confidence > 70 else is_ai

            return {
                'is_ai': is_ai,
                'confidence': float(confidence),
                'method': 'huggingface',
                'raw_label': result[0]['label']
            }
        except Exception as e:
            print(f"❌ Error in HuggingFace prediction: {e}")
            return None
    
    def predict_custom(self, image_path):
        """Predict using custom trained model"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                # Assuming class 0 = Real, class 1 = AI
                is_ai = predicted.item() == 1
                conf = confidence.item() * 100
                
                return {
                    'is_ai': is_ai,
                    'confidence': float(conf),
                    'method': 'custom_model'
                }
        except Exception as e:
            print(f"❌ Error in custom prediction: {e}")
            return None

    def _get_jpeg_qtable(self, image_path):
        """Return flattened JPEG quantization table values as numpy array if JPEG, else None"""
        try:
            from PIL import Image
            im = Image.open(image_path)
            if im.format != 'JPEG':
                return None
            qtables = im.quantization
            vals = []
            if isinstance(qtables, dict):
                for k, v in qtables.items():
                    if isinstance(v, (list, tuple)):
                        vals.extend(v)
                    else:
                        vals.append(v)
            elif isinstance(qtables, (list, tuple)):
                for v in qtables:
                    if isinstance(v, (list, tuple)):
                        vals.extend(v)
                    else:
                        vals.append(v)
            return np.array(vals, dtype=float) if vals else None
        except Exception:
            return None

    def predict(self, image_path):
        """
        Main prediction method - Works on ANY image type
        
        Args:
            image_path: Path to image (laptop, mobile, chair, car, etc.)
        
        Returns:
            dict with detection results
        """
        results = {
            'image_path': image_path,
            'is_ai_generated': False,
            'confidence': 0.0,
            'label': 'Real Photo',
            'method': self.method,
            'verdict': '',
            'explanation': '',
            'metrics': {}
        }
        
        try:
            if self.method == 'huggingface':
                hf_result = self.predict_huggingface(image_path)
                if hf_result:
                    results['is_ai_generated'] = hf_result['is_ai']
                    results['confidence'] = hf_result['confidence']
                    results['label'] = 'AI Generated' if hf_result['is_ai'] else 'Real Photo'
            
            elif self.method == 'custom':
                custom_result = self.predict_custom(image_path)
                if custom_result:
                    results['is_ai_generated'] = custom_result['is_ai']
                    results['confidence'] = custom_result['confidence']
                    results['label'] = 'AI Generated' if custom_result['is_ai'] else 'Real Photo'

                    # When user requested high sensitivity, do an additional artifact analysis
                    # and override to AI if artifact score is strongly suspicious
                    if self.sensitivity == 'high':
                        try:
                            artifact_result = self.analyze_artifacts(image_path)
                            if artifact_result and artifact_result.get('metrics', {}).get('ai_score', 0) >= 55:
                                # Under high sensitivity, give priority to artifact score and be decisive
                                art_score = artifact_result['metrics']['ai_score']
                                results['is_ai_generated'] = True
                                # Use artifact ai_score directly as confidence (more decisive)
                                results['confidence'] = float(min(100, art_score))
                                results['label'] = 'AI Generated'
                                results['metrics'] = artifact_result.get('metrics', {})
                                results['explanation'] = f"Artifact override: {art_score:.1f}% ai_score"
                        except Exception:
                            pass
            
            elif self.method == 'artifact':
                artifact_result = self.analyze_artifacts(image_path)
                if artifact_result:
                    results['is_ai_generated'] = artifact_result['is_ai']
                    results['confidence'] = artifact_result['confidence']
                    results['label'] = 'AI Generated' if artifact_result['is_ai'] else 'Real Photo'
                    results['metrics'] = artifact_result.get('metrics', {})
                    results['explanation'] = ' | '.join(artifact_result.get('explanations', []))
            
            elif self.method == 'hybrid':
                # Try HuggingFace first
                hf_result = self.predict_huggingface(image_path)
                artifact_result = self.analyze_artifacts(image_path)
                
                # Tuneable weighting - favor artifacts more under high sensitivity or if local model missing
                hf_weight = 0.7
                artifact_weight = 0.3
                decision_threshold = 50
                if self.sensitivity == 'high':
                    # Balanced weighting for high sensitivity - don't over-rely on artifacts
                    artifact_weight = 0.4
                    hf_weight = 0.6
                    decision_threshold = 45

                elif hf_result and artifact_result:
                    # Use weighted scoring and normalize
                    hf_score = hf_result['confidence'] if hf_result['is_ai'] else (100 - hf_result['confidence'])
                    artifact_score = artifact_result['confidence'] if artifact_result['is_ai'] else (100 - artifact_result['confidence'])
                    
                    # ARTIFACT OVERRIDE DISABLED
                    # The override was causing false positives on high-quality real images (cars, keyboards).
                    # We now rely primarily on the Deep Learning model.
                    # artifact_ai_score = artifact_result.get('metrics', {}).get('ai_score', 0)
                    # if artifact_ai_score > 75: ...

                    # Tuneable weighting - Reduced artifact influence to avoid false positives
                    hf_weight = 0.8
                    artifact_weight = 0.2
                    decision_threshold = 50
                    if self.sensitivity == 'high':
                        # Even in high sensitivity, keeps artifacts subordinate to the model
                        artifact_weight = 0.3
                        hf_weight = 0.7
                        decision_threshold = 45

                    combined_score = (hf_weight * hf_score + artifact_weight * artifact_score)
                    is_ai = combined_score >= decision_threshold
                    
                    results['is_ai_generated'] = is_ai
                    results['confidence'] = combined_score if is_ai else (100 - combined_score)
                    results['label'] = 'AI Generated' if is_ai else 'Real Photo'
                    results['metrics'] = artifact_result.get('metrics', {})
                    results['explanation'] = f"HuggingFace: {hf_result['confidence']:.1f}% | Artifacts: {artifact_result['confidence']:.1f}%"
                
                elif hf_result:
                    results['is_ai_generated'] = hf_result['is_ai']
                    results['confidence'] = hf_result['confidence']
                    results['label'] = 'AI Generated' if hf_result['is_ai'] else 'Real Photo'
                
                elif artifact_result:
                    results['is_ai_generated'] = artifact_result['is_ai']
                    results['confidence'] = artifact_result['confidence']
                    results['label'] = 'AI Generated' if artifact_result['is_ai'] else 'Real Photo'
                    results['metrics'] = artifact_result.get('metrics', {})
            
            # Generate verdict message
            # Make verdict messages more decisive under high sensitivity for AI detection
            if results['is_ai_generated']:
                if results['confidence'] > 80:
                    results['verdict'] = f"⚠️ This image is LIKELY AI-GENERATED with {results['confidence']:.1f}% confidence"
                elif results['confidence'] > 60:
                    results['verdict'] = f"⚠️ This image APPEARS to be AI-generated ({results['confidence']:.1f}% confidence)"
                else:
                    results['verdict'] = f"❓ Uncertain, but leans towards AI-generated ({results['confidence']:.1f}% confidence)"
            else:
                # If the user requested high sensitivity and the detector is 'custom' or 'hybrid', be slightly stricter when labeling as real
                if self.sensitivity == 'high' and results['confidence'] < 85 and self.method in ['custom','hybrid']:
                    results['verdict'] = f"❓ Low confidence real photo ({results['confidence']:.1f}%) — consider retraining with more AI samples"
                elif results['confidence'] > 80:
                    results['verdict'] = f"✅ This image is LIKELY A REAL PHOTO with {results['confidence']:.1f}% confidence"
                elif results['confidence'] > 60:
                    results['verdict'] = f"✅ This image APPEARS to be a real photo ({results['confidence']:.1f}% confidence)"
                else:
                    results['verdict'] = f"❓ Uncertain, but leans towards real photo ({results['confidence']:.1f}% confidence)"
            
            return results
        
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            results['verdict'] = f"❌ Error analyzing image: {str(e)}"
            return results


# ═══════════════════════════════════════════════════════════════
# 🧪 TEST THE DETECTOR
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING AI IMAGE DETECTOR")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = AIImageDetector(method='artifact')  # Use 'artifact' if HF not available
    
    # Test on a sample image
    test_image = "data/images/test/laptop/test_image.jpg"  # Replace with actual path
    
    print(f"📸 Analyzing: {test_image}\n")
    result = detector.predict(test_image)
    
    print("🎯 RESULTS:")
    print("-" * 60)
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Verdict: {result['verdict']}")
    print(f"Method: {result['method']}")
    
    if result.get('metrics'):
        print("\n📊 Technical Metrics:")
        for key, value in result['metrics'].items():
            print(f"  - {key}: {value:.4f}")
    
    if result.get('explanation'):
        print(f"\n💡 Explanation: {result['explanation']}")
    
    print("\n" + "="*60 + "\n")