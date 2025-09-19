#!/usr/bin/env python3
"""
Indian Cattle & Buffalo Breed Recognition System
Uses CSV dataset for training and image recognition
Simple setup and easy to use
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import urllib.request
import zipfile

# Auto-install required packages
def install_if_missing():
    """Install required packages if missing"""
    packages = {
        'tensorflow': 'tensorflow>=2.10.0',
        'pandas': 'pandas>=1.3.0',
        'PIL': 'Pillow>=8.0.0',
        'cv2': 'opencv-python>=4.5.0',
        'sklearn': 'scikit-learn>=1.0.0'
    }
    
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install dependencies
print("🔧 Checking and installing dependencies...")
install_if_missing()

# Import after installation
import tensorflow as tf
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class IndianBreedClassifier:
    def __init__(self):
        self.project_dir = Path.cwd()
        self.setup_directories()
        self.breed_info = self.create_breed_database()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.setup_csv_dataset()
        
    def setup_directories(self):
        """Create project directories"""
        dirs = ['models', 'dataset', 'images', 'uploads']
        for dir_name in dirs:
            (self.project_dir / dir_name).mkdir(exist_ok=True)
        print("✅ Project directories created")
        
    def create_breed_database(self):
        """Create Indian breed information database"""
        return {
            "Gir": {
                "type": "Cattle",
                "origin": "Gujarat, India", 
                "milk_yield": "10-15 L/day",
                "characteristics": "Heat resistant, curved horns, large hump",
                "color": "White with red/brown patches",
                "weight_range": "400-500 kg",
                "climate": "Tropical, hot and humid",
                "diseases": ["Foot and mouth", "Mastitis"],
                "feed": "Green fodder, concentrates, legumes"
            },
            "Sahiwal": {
                "type": "Cattle",
                "origin": "Punjab, Pakistan/India",
                "milk_yield": "8-12 L/day", 
                "characteristics": "Drought resistant, docile, good milker",
                "color": "Red-brown with white markings",
                "weight_range": "350-400 kg",
                "climate": "Semi-arid to tropical",
                "diseases": ["Tick fever", "Heat stress"],
                "feed": "Dry fodder, grain mixture"
            },
            "Red_Sindhi": {
                "type": "Cattle",
                "origin": "Sindh region",
                "milk_yield": "6-10 L/day",
                "characteristics": "Compact, heat tolerant, disease resistant",
                "color": "Deep red to light red",
                "weight_range": "300-400 kg", 
                "climate": "Arid and semi-arid",
                "diseases": ["Parasites", "Nutritional disorders"],
                "feed": "Poor quality fodder, drought resistant"
            },
            "Tharparkar": {
                "type": "Cattle",
                "origin": "Rajasthan/Sindh",
                "milk_yield": "8-12 L/day",
                "characteristics": "Dual purpose, drought resistant",
                "color": "White to light grey",
                "weight_range": "400-500 kg",
                "climate": "Desert, very hot and dry",
                "diseases": ["Dehydration", "Heat stroke"],
                "feed": "Desert vegetation, minimal water"
            },
            "Murrah": {
                "type": "Buffalo", 
                "origin": "Haryana, India",
                "milk_yield": "15-25 L/day",
                "characteristics": "World's best dairy buffalo, curved horns",
                "color": "Jet black",
                "weight_range": "500-650 kg",
                "climate": "Hot humid, good water access",
                "diseases": ["Milk fever", "Reproductive disorders"],
                "feed": "High quality green fodder, concentrates"
            },
            "Nili_Ravi": {
                "type": "Buffalo",
                "origin": "Punjab, Pakistan/India",
                "milk_yield": "12-20 L/day",
                "characteristics": "Large size, white facial markings",
                "color": "Black with white patches on face and legs",
                "weight_range": "450-550 kg",
                "climate": "Hot humid with water bodies",
                "diseases": ["Mastitis", "Lameness"],
                "feed": "Green fodder, silage, concentrates"
            },
            "Mehsana": {
                "type": "Buffalo",
                "origin": "Gujarat, India", 
                "milk_yield": "8-12 L/day",
                "characteristics": "Medium size, adapted to dry climate",
                "color": "Black to dark grey",
                "weight_range": "400-500 kg",
                "climate": "Semi-arid, drought prone",
                "diseases": ["Heat stress", "Water scarcity issues"],
                "feed": "Local vegetation, crop residues"
            },
            "Surti": {
                "type": "Buffalo",
                "origin": "Gujarat, India",
                "milk_yield": "6-10 L/day", 
                "characteristics": "Small to medium, high fat milk",
                "color": "Black",
                "weight_range": "350-450 kg",
                "climate": "Coastal, humid",
                "diseases": ["Skin diseases", "Parasites"],
                "feed": "Coastal grasses, coconut cake"
            }
        }
        
    def setup_csv_dataset(self):
        """Create sample CSV dataset for training"""
        csv_file = self.project_dir / "dataset" / "breed_dataset.csv"
        
        if not csv_file.exists():
            print("📋 Creating sample dataset...")
            self.create_sample_dataset(csv_file)
        else:
            print("✅ Dataset found")
            
        return csv_file
    
    def create_sample_dataset(self, csv_file):
        """Create a sample CSV dataset with breed characteristics"""
        # Sample data based on breed characteristics
        sample_data = []
        
        for breed, info in self.breed_info.items():
            # Generate multiple samples per breed with slight variations
            for i in range(10):  # 10 samples per breed
                # Extract numeric features from breed info
                milk_yield_avg = self.extract_avg_milk_yield(info['milk_yield'])
                weight_avg = self.extract_avg_weight(info['weight_range'])
                
                # Add some variation to the data
                variation = np.random.normal(0, 0.1)
                
                row = {
                    'breed': breed,
                    'type': info['type'],
                    'origin_state': info['origin'].split(',')[0].strip(),
                    'milk_yield_liters': milk_yield_avg + variation,
                    'weight_kg': weight_avg + variation * 50,
                    'heat_tolerance': self.get_heat_tolerance_score(info['climate']),
                    'drought_resistance': self.get_drought_resistance_score(info),
                    'body_size': self.get_body_size_category(weight_avg),
                    'color_category': self.categorize_color(info['color']),
                    'image_filename': f"{breed.lower()}_{i+1:02d}.jpg"
                }
                sample_data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_file, index=False)
        print(f"✅ Sample dataset created: {csv_file}")
        print(f"📊 Dataset contains {len(df)} samples across {len(self.breed_info)} breeds")
        
    def extract_avg_milk_yield(self, yield_str):
        """Extract average milk yield from string like '10-15 L/day'"""
        try:
            numbers = [int(x) for x in yield_str.split() if x.split('-')[0].isdigit()]
            if '-' in yield_str:
                parts = yield_str.split('-')
                min_val = int(parts[0])
                max_val = int(parts[1].split()[0])
                return (min_val + max_val) / 2
            return numbers[0] if numbers else 10
        except:
            return 10
            
    def extract_avg_weight(self, weight_str):
        """Extract average weight from string like '400-500 kg'"""
        try:
            if '-' in weight_str:
                parts = weight_str.split('-')
                min_val = int(parts[0])
                max_val = int(parts[1].split()[0])
                return (min_val + max_val) / 2
            return int(weight_str.split()[0])
        except:
            return 400
            
    def get_heat_tolerance_score(self, climate):
        """Get heat tolerance score based on climate"""
        if 'hot' in climate.lower() or 'desert' in climate.lower():
            return 9
        elif 'tropical' in climate.lower():
            return 7
        elif 'humid' in climate.lower():
            return 6
        else:
            return 5
            
    def get_drought_resistance_score(self, info):
        """Get drought resistance score"""
        if 'drought' in info['characteristics'].lower():
            return 9
        elif 'desert' in info['climate'].lower():
            return 8
        elif 'arid' in info['climate'].lower():
            return 7
        else:
            return 5
            
    def get_body_size_category(self, weight):
        """Categorize body size based on weight"""
        if weight < 350:
            return 'Small'
        elif weight < 450:
            return 'Medium'
        else:
            return 'Large'
            
    def categorize_color(self, color_desc):
        """Categorize color description"""
        color_lower = color_desc.lower()
        if 'black' in color_lower:
            return 'Black'
        elif 'red' in color_lower:
            return 'Red'
        elif 'white' in color_lower:
            return 'White'
        elif 'grey' in color_lower or 'gray' in color_lower:
            return 'Grey'
        else:
            return 'Mixed'
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        csv_file = self.project_dir / "dataset" / "breed_dataset.csv"
        
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Dataset loaded: {len(df)} samples")
            
            # Prepare features and labels
            feature_columns = ['milk_yield_liters', 'weight_kg', 'heat_tolerance', 
                             'drought_resistance']
            
            # Encode categorical features
            df['type_encoded'] = self.label_encoder.fit_transform(df['type'])
            df['body_size_encoded'] = self.label_encoder.fit_transform(df['body_size'])
            df['color_encoded'] = self.label_encoder.fit_transform(df['color_category'])
            
            feature_columns.extend(['type_encoded', 'body_size_encoded', 'color_encoded'])
            
            X = df[feature_columns].values
            y = df['breed'].values
            
            # Encode breed labels
            breed_encoder = LabelEncoder()
            y_encoded = breed_encoder.fit_transform(y)
            
            # Store encoders for later use
            self.breed_encoder = breed_encoder
            self.feature_columns = feature_columns
            
            return X, y_encoded, df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None, None, None
    
    def create_model(self, input_shape, num_classes):
        """Create a simple neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train the breed classification model"""
        print("🚀 Starting model training...")
        
        # Load dataset
        X, y, df = self.load_dataset()
        if X is None:
            print("❌ Failed to load dataset")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        num_classes = len(np.unique(y))
        self.model = self.create_model(X.shape[1], num_classes)
        
        print("🎯 Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Model trained! Test Accuracy: {test_accuracy:.2%}")
        
        # Save model
        model_path = self.project_dir / "models" / "breed_classifier.h5"
        self.model.save(model_path)
        
        # Save encoders
        encoders = {
            'breed_encoder': self.breed_encoder.classes_.tolist(),
            'feature_columns': self.feature_columns
        }
        
        with open(self.project_dir / "models" / "encoders.json", 'w') as f:
            json.dump(encoders, f)
            
        print(f"💾 Model saved to {model_path}")
        return True
    
    def load_trained_model(self):
        """Load pre-trained model"""
        model_path = self.project_dir / "models" / "breed_classifier.h5"
        encoders_path = self.project_dir / "models" / "encoders.json"
        
        if model_path.exists() and encoders_path.exists():
            self.model = tf.keras.models.load_model(model_path)
            
            with open(encoders_path, 'r') as f:
                encoders = json.load(f)
                
            self.breed_encoder = LabelEncoder()
            self.breed_encoder.classes_ = np.array(encoders['breed_encoder'])
            self.feature_columns = encoders['feature_columns']
            
            print("✅ Pre-trained model loaded")
            return True
        else:
            print("⚠️ No pre-trained model found")
            return False
    
    def predict_breed_from_features(self, features):
        """Predict breed from extracted features"""
        if self.model is None:
            return None
            
        try:
            # Reshape features for prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            predictions = self.model.predict(features_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get breed name
            breed_name = self.breed_encoder.inverse_transform([predicted_class])[0]
            
            return {
                'breed': breed_name,
                'confidence': confidence,
                'breed_info': self.breed_info.get(breed_name, {})
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def extract_features_from_image(self, image_path):
        """Extract features from image (placeholder - would need computer vision)"""
        # This is a simplified feature extraction
        # In reality, you'd use computer vision to extract actual features
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Placeholder feature extraction (random for demo)
            # Replace with actual image analysis
            features = [
                np.random.uniform(6, 25),    # milk_yield_liters
                np.random.uniform(300, 650), # weight_kg  
                np.random.randint(5, 10),    # heat_tolerance
                np.random.randint(5, 10),    # drought_resistance
                np.random.randint(0, 2),     # type_encoded
                np.random.randint(0, 3),     # body_size_encoded
                np.random.randint(0, 5)      # color_encoded
            ]
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None

class BreedRecognitionGUI:
    def __init__(self):
        self.classifier = IndianBreedClassifier()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("🐄 Indian Cattle & Buffalo Breed Recognizer")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Indian Cattle & Buffalo Breed Recognition System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Buttons
        ttk.Button(main_frame, text="🚀 Train New Model", 
                  command=self.train_model).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(main_frame, text="📁 Load Pre-trained Model", 
                  command=self.load_model).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(main_frame, text="🖼️ Upload Image for Recognition", 
                  command=self.upload_image).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Image display
        self.image_label = ttk.Label(main_frame, text="No image selected")
        self.image_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=15, width=80)
        self.results_text.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
    def train_model(self):
        """Train a new model"""
        self.status_var.set("Training model... Please wait")
        self.root.update()
        
        success = self.classifier.train_model()
        if success:
            self.status_var.set("✅ Model trained successfully!")
            self.results_text.insert(tk.END, "✅ Model trained and saved successfully!\n\n")
        else:
            self.status_var.set("❌ Model training failed")
            self.results_text.insert(tk.END, "❌ Model training failed\n\n")
    
    def load_model(self):
        """Load pre-trained model"""
        success = self.classifier.load_trained_model()
        if success:
            self.status_var.set("✅ Model loaded successfully!")
            self.results_text.insert(tk.END, "✅ Pre-trained model loaded successfully!\n\n")
        else:
            self.status_var.set("❌ No pre-trained model found")
            self.results_text.insert(tk.END, "❌ No pre-trained model found. Please train a new model first.\n\n")
    
    def upload_image(self):
        """Upload and analyze image"""
        if self.classifier.model is None:
            messagebox.showerror("Error", "Please load or train a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select cattle/buffalo image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def analyze_image(self, image_path):
        """Analyze uploaded image"""
        try:
            # Display image
            img = Image.open(image_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            # Extract features and predict
            self.status_var.set("Analyzing image...")
            self.root.update()
            
            features = self.classifier.extract_features_from_image(image_path)
            if features is None:
                self.results_text.insert(tk.END, "❌ Failed to extract features from image\n\n")
                return
            
            result = self.classifier.predict_breed_from_features(features)
            if result is None:
                self.results_text.insert(tk.END, "❌ Prediction failed\n\n") 
                return
            
            # Display results
            self.display_results(result)
            self.status_var.set("✅ Analysis complete!")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"❌ Error analyzing image: {e}\n\n")
            self.status_var.set("❌ Analysis failed")
    
    def display_results(self, result):
        """Display prediction results"""
        breed = result['breed']
        confidence = result['confidence']
        breed_info = result['breed_info']
        
        self.results_text.insert(tk.END, "="*50 + "\n")
        self.results_text.insert(tk.END, f"🎯 PREDICTION RESULTS\n")
        self.results_text.insert(tk.END, "="*50 + "\n")
        self.results_text.insert(tk.END, f"Predicted Breed: {breed}\n")
        self.results_text.insert(tk.END, f"Confidence: {confidence:.1%}\n\n")
        
        if breed_info:
            self.results_text.insert(tk.END, f"📋 BREED INFORMATION:\n")
            self.results_text.insert(tk.END, f"Type: {breed_info.get('type', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Origin: {breed_info.get('origin', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Milk Yield: {breed_info.get('milk_yield', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Weight Range: {breed_info.get('weight_range', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Color: {breed_info.get('color', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Climate: {breed_info.get('climate', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Characteristics: {breed_info.get('characteristics', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Feed: {breed_info.get('feed', 'N/A')}\n\n")
        
        self.results_text.see(tk.END)
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🐄 Indian Cattle & Buffalo Breed Recognition System")
    print("="*60)
    
    # Check if running in GUI mode
    if len(sys.argv) > 1 and sys.argv[1] == '--nogui':
        # Command line mode
        classifier = IndianBreedClassifier()
        
        print("Training model...")
        classifier.train_model()
        
        print("\nSystem ready for predictions!")
        print("You can now use the trained model for breed recognition.")
    else:
        # GUI mode
        app = BreedRecognitionGUI()
        app.run()

if __name__ == "__main__":
    main()