import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import warnings
import argparse
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# For efficient training
from lightgbm import LGBMClassifier

class SteganographyDetector:
    def __init__(self, clean_dir, stego_dir, model_save_path="stego_detector_model.pkl"):
        self.clean_dir = clean_dir
        self.stego_dir = stego_dir
        self.model_save_path = model_save_path
        self.model = None
        self.feature_names = []
        
    def extract_features(self, img_path):
        """Extract statistical features from image that can help detect steganography"""
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            features = []
            feature_names = []
            
            # Process each color channel
            for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
                channel = img_array[:, :, channel_idx].flatten()
                
                # 1. LSB-specific statistics
                lsb = channel % 2  
                lsb_ratio = np.sum(lsb) / len(lsb)  
                
                # 2. LSB transitions (detecting unnatural patterns)
                lsb_transitions = np.sum(np.abs(np.diff(lsb))) / (len(lsb) - 1) if len(lsb) > 1 else 0
                
                # 3. Pairs analysis (looking at adjacent pixel relationships)
                pairs = np.abs(np.diff(channel))
                even_pairs = np.mean(pairs[channel[:-1] % 2 == 0])
                odd_pairs = np.mean(pairs[channel[:-1] % 2 == 1])
                pair_ratio = even_pairs / odd_pairs if odd_pairs > 0 else 0
                
                # 4. Statistical moments
                mean = np.mean(channel)
                std = np.std(channel)
                skewness = np.mean(((channel - mean) / (std + 1e-10)) ** 3) if std > 0 else 0
                kurtosis = np.mean(((channel - mean) / (std + 1e-10)) ** 4) - 3 if std > 0 else 0
                
                # 5. Histogram analysis
                hist, _ = np.histogram(channel, bins=256, range=[0, 256])
                hist = hist / np.sum(hist)  # Normalize
                
                # Common LSB steganography affects even/odd balance
                even_bins = np.sum(hist[::2])
                odd_bins = np.sum(hist[1::2])
                bin_ratio = even_bins / odd_bins if odd_bins > 0 else 0
                
                # 6. Entropy-related features
                non_zero_hist = hist[hist > 0]
                entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist)) if len(non_zero_hist) > 0 else 0
                
                # 7. Co-occurrence analysis
                if img_array.shape[0] > 1 and img_array.shape[1] > 1:
                    channel_2d = channel.reshape(img_array.shape[0], img_array.shape[1])
                    # Horizontal, vertical, and diagonal pairs
                    h_diff = np.abs(channel_2d[:, :-1] - channel_2d[:, 1:]).mean()
                    v_diff = np.abs(channel_2d[:-1, :] - channel_2d[1:, :]).mean()
                    d_diff = np.abs(channel_2d[:-1, :-1] - channel_2d[1:, 1:]).mean()
                else:
                    h_diff, v_diff, d_diff = 0, 0, 0
                
                # 8. LSB correlation
                if len(lsb) > 1:
                    # Correlation between consecutive LSBs
                    lsb_correlation = np.corrcoef(lsb[:-1], lsb[1:])[0, 1] if len(np.unique(lsb)) > 1 else 0
                else:
                    lsb_correlation = 0
                
                # 9. LSB planes analysis
                # Create 8 bit planes
                bit_planes = [(channel >> i) & 1 for i in range(8)]
                # Calculate correlation between LSB and other bit planes
                bit_corrs = []
                for i in range(1, 8):
                    if len(np.unique(bit_planes[0])) > 1 and len(np.unique(bit_planes[i])) > 1:
                        corr = np.abs(np.corrcoef(bit_planes[0], bit_planes[i])[0, 1])
                        bit_corrs.append(corr)
                    else:
                        bit_corrs.append(0)
                lsb_higher_correlation = np.mean(bit_corrs) if bit_corrs else 0
                
                # 10. Add selected histogram bins (using fewer bins to reduce dimensions)
                bin_indices = [0, 32, 64, 96, 128, 160, 192, 224, 255]
                hist_features = [hist[i] for i in bin_indices]
                
                # Combine all features
                channel_features = [
                    mean, std, skewness, kurtosis,
                    lsb_ratio, lsb_transitions, lsb_correlation, lsb_higher_correlation,
                    pair_ratio, bin_ratio, entropy,
                    h_diff, v_diff, d_diff
                ]
                channel_features.extend(hist_features)
                features.extend(channel_features)
                
                # Create feature names if needed
                if not self.feature_names:
                    base_names = [
                        f"{channel_name}_mean", f"{channel_name}_std", 
                        f"{channel_name}_skewness", f"{channel_name}_kurtosis",
                        f"{channel_name}_lsb_ratio", f"{channel_name}_lsb_transitions", 
                        f"{channel_name}_lsb_correlation", f"{channel_name}_lsb_higher_correlation",
                        f"{channel_name}_pair_ratio", f"{channel_name}_bin_ratio", f"{channel_name}_entropy",
                        f"{channel_name}_h_diff", f"{channel_name}_v_diff", f"{channel_name}_d_diff"
                    ]
                    hist_names = [f"{channel_name}_hist_bin_{i}" for i in bin_indices]
                    feature_names.extend(base_names + hist_names)
            
            if not self.feature_names and features:
                self.feature_names = feature_names
                
            return features
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return a vector of zeros in case of error
            if self.feature_names:
                return [0] * len(self.feature_names)
            else:
                # Estimate feature count and return zeros
                return [0] * 69  # 23 features per channel * 3 channels
    
    def process_image(self, args):
        """Helper function for parallel processing"""
        img_path, is_stego = args
        try:
            features = self.extract_features(img_path)
            return features, is_stego
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, is_stego
    
    def prepare_dataset(self):
        """Prepare features and labels from image directories using parallel processing"""
        # Gather all file paths
        clean_files = [os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        stego_files = [os.path.join(self.stego_dir, f) for f in os.listdir(self.stego_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Create task list with labels (0 for clean, 1 for stego)
        tasks = [(f, 0) for f in clean_files] + [(f, 1) for f in stego_files]
        
        print(f"Processing {len(clean_files)} clean images and {len(stego_files)} stego images...")
        
        # Process files in parallel
        n_workers = min(cpu_count(), 8)  # Use up to 8 cores
        results = []
        
        with Pool(n_workers) as pool:
            for result in tqdm(pool.imap(self.process_image, tasks), total=len(tasks)):
                results.append(result)
        
        # Extract features and labels
        features = []
        labels = []
        
        for feature, label in results:
            if feature is not None:
                features.append(feature)
                labels.append(label)
        
        # If we haven't processed any files yet, extract features from one to get feature names
        if not self.feature_names and tasks:
            self.extract_features(tasks[0][0])
            
        # Create DataFrame
        X = pd.DataFrame(features, columns=self.feature_names)
        y = np.array(labels)
        
        # Verify data is good
        if len(X) == 0:
            raise ValueError("No valid features extracted from images")
            
        print(f"Successfully processed {len(X)} images")
        return X, y
    
    def train(self, test_size=0.2, random_state=42):
        """Train the steganography detection model"""
        start_time = time.time()
        
        print("Preparing dataset...")
        X, y = self.prepare_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        # Train model (LightGBM for speed and accuracy)
        self.model = LGBMClassifier(
            n_estimators=200,  # Increased for better accuracy
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,  # Limit depth to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,  # Use all available cores
            random_state=random_state,
            verbose=-1  # Quiet mode
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save model
        joblib.dump(self.model, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Clean', 'Steganography']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Plot feature importance
        top_features = feature_importance.head(15)
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': train_time
        }
    
    def predict(self, img_path):
        """Predict whether an image contains steganography"""
        if self.model is None:
            if os.path.exists(self.model_save_path):
                self.model = joblib.load(self.model_save_path)
            else:
                raise ValueError("Model not trained. Call train() first.")
        
        features = self.extract_features(img_path)
        features_df = pd.DataFrame([features], columns=self.feature_names)
        
        prediction = self.model.predict(features_df)[0]
        probability = self.model.predict_proba(features_df)[0][1]
        
        return {
            'is_stego': bool(prediction),
            'probability': probability
        }


def main():
    parser = argparse.ArgumentParser(description='Train an improved steganography detection model')
    parser.add_argument('--clean', type=str, default='clean', help='Directory with clean images')
    parser.add_argument('--stego', type=str, default='lsb', help='Directory with steganographic images')
    parser.add_argument('--model', type=str, default='stego_detector_model.pkl', help='Path to save model')
    parser.add_argument('--test', type=str, help='Path to a test image (optional)')
    
    args = parser.parse_args()
    
    print(f"Training model using clean images from '{args.clean}' and stego images from '{args.stego}'")
    
    detector = SteganographyDetector(args.clean, args.stego, args.model)
    results = detector.train()
    
    print("\nResults Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    
    # Test on a specific image if provided
    if args.test:
        if os.path.exists(args.test):
            print(f"\nTesting on image: {args.test}")
            result = detector.predict(args.test)
            print(f"Is steganography detected? {result['is_stego']}")
            print(f"Probability: {result['probability']:.4f}")
        else:
            print(f"Test image not found: {args.test}")
    
    print("\nDone! Model saved to", args.model)


if __name__ == "__main__":
    main()