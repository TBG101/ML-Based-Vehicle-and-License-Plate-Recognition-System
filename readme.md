# 🚗🔍 ML-Based Vehicle and License Plate Recognition System

## 📘 Project Description
This university project aims to develop an intelligent system for vehicle and license plate recognition using machine learning. Leveraging cutting-edge tools like YOLOv8 and TensorFlow, the system integrates a secure backend with Flask and PostgreSQL for data management.

## ✨ Features  
- 🧠 **License Plate Detection** using **YOLOv8**  
- 🚙 **Vehicle Type Classification** with a custom **TensorFlow/Keras** model  
- 🔐 **User Authentication & Authorization** with **JWT**  
- 🌐 **RESTful API** for predictions and user management  
- 🐳 **Dockerized PostgreSQL Database** for persistent storage  

## ⚙️ Setup

### ✅ Prerequisites  
- 🐍 Python 3.8 or higher  
- 🐳 Docker & Docker Compose  
- 💡 (Optional) Virtual Environment  


### 📦 Installation  
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd poly_project/4eme/ml/backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the `backend` directory.
   - Add the following variables:
     ```env
     ROBOFLOW_API_KEY=<your_roboflow_api_key>
     JWT_SECRET=<your_jwt_secret>
     DATABASE_URL=postgresql://yourusername:yourpassword@localhost:5432/yourdatabase
     ```
4. Start the PostgreSQL database:
   ```bash
   docker-compose up -d
   ```
5. Run the Flask server:
   ```bash
   python server.py
   ```

## 🧪 Usage

### 🏋️‍♂️ Training Models
- **📷 License Plate Recognition**: Run train_model.py in the plateRecognition directory to train YOLO.
- **🚘 Vehicle Classification**: Run `vehicle_classifier_training.py` in the `carRecognition` directory to train the TensorFlow model.

| Endpoint                         | Method | Description                            |
|----------------------------------|--------|----------------------------------------|
| `/api/v1/health`                | GET    | ✅ Health Check                        |
| `/api/v1/signup`                | POST   | 📝 User Signup                         |
| `/api/v1/login`                 | POST   | 🔐 User Login                          |
| `/api/v1/me`                    | GET    | 👤 Get User Info                       |
| `/api/v1/predict`              | POST   | 🔍 Predict License Plate & Vehicle Type |
| `/api/v1/uploads/<filename>`   | GET    | 🖼 Retrieve Uploaded Image             |


## 📥 Example Request
To predict license plates and vehicle types:
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Authorization: Bearer <your_jwt_token>" \
  -F "file=@/path/to/your/image.jpg"
```
   

## Contributors
    **TODO**