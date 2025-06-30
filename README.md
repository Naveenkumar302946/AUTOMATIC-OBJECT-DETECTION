# AUTOMATIC-OBJECT-DETECTION
  This project aims to develop an advanced image segmentation tool designed to automatically detect and delineate objects in satellite imagery using deep learning techniques. 
# 🌍 U-Net Semantic Segmentation on Aerial (Satellite) Imagery

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20M1-lightgrey)](https://developer.apple.com/silicon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project uses a **U-Net** architecture for semantic segmentation on high-resolution **aerial or satellite imagery**. The goal is to identify and classify different land cover types such as **buildings, roads, vegetation, and water bodies** at the pixel level.

---

## 📁 Project Structure

U-Net-Semantic-Segmentation-Aerial-Images/
├── imagesegemntation.py # Inference script using trained model
├── preprocess.py # Convert image/mask datasets to NumPy arrays
├── train_unet.py # Train U-Net model
├── predict.py # Predict and visualize results
├── dataset/
│ ├── images/ # Input images
│ └── masks/ # Ground truth masks
├── models/ # Saved model weights
├── X.npy / y.npy # Preprocessed datasets
├── requirements.txt # Python dependencies
└── README.md # You're here!

yaml
Copy
Edit

---

## 🛠️ Environment Setup (MacBook Air M1)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/U-Net-Semantic-Segmentation-Aerial-Images.git
   cd U-Net-Semantic-Segmentation-Aerial-Images
Create and Activate Conda Environment

bash
Copy
Edit
conda create -n unet_segmentation python=3.9
conda activate unet_segmentation
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
(Optional for PyTorch on M1 with GPU acceleration)

bash
Copy
Edit
pip install torch torchvision torchaudio
🔄 Data Preprocessing
Ensure your image and mask data are placed inside the dataset/images and dataset/masks folders, respectively.

Then run:

bash
Copy
Edit
python preprocess.py
This will generate X.npy and y.npy files used for training.

🧠 Train the U-Net Model
Train the model using:

bash
Copy
Edit
python train_unet.py
You can adjust image size, batch size, and number of classes inside train_unet.py.

📈 Make Predictions
After training, run predictions on test images:

bash
Copy
Edit
python predict.py
You can also try the full pipeline from preprocessing to prediction.

🎨 Output Sample
Original Image	Ground Truth Mask	Predicted Mask

(Make sure to replace these with your actual output images)

📦 Dependencies
Python 3.9

NumPy

OpenCV

TensorFlow or PyTorch

Matplotlib

tqdm

scikit-learn

📌 To-Do
 Add color mapping for better visual mask outputs

 Integrate with React-based web frontend

 Add support for webcam/live satellite input

👨‍💻 Author
Naveen Kumar V
B.E. Computer Science and Engineering (2021–2025)
Rajalakshmi Institute of Technology, Chennai

