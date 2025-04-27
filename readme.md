# 🚗🔍 ML-Based Vehicle and License Plate Recognition System

## 📘 Project Description
This university project at Ecole Polytechnique Sousse aims to develop an intelligent system for vehicle and license plate recognition using machine learning. Leveraging cutting-edge tools like YOLOv8 and TensorFlow, the system integrates a secure backend with Flask and PostgreSQL for data management.

---


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
   git clone https://github.com/TBG101/ML-Based-Vehicle-and-License-Plate-Recognition-System
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

### 🏋️️ Training Models
- **📷 License Plate Recognition**: Run `train_model.py` in the `plateRecognition` directory to train YOLO.
- **🚘 Vehicle Classification**: Run `vehicle_classifier_training.py` in the `carRecognition` directory to train the TensorFlow model.

| Endpoint                         | Method | Description                            |
|----------------------------------|--------|----------------------------------------|
| `/api/v1/health`                | GET    | ✅ Health Check                        |
| `/api/v1/signup`                | POST   | 📝 User Signup                         |
| `/api/v1/login`                 | POST   | 🔐 User Login                          |
| `/api/v1/me`                    | GET    | 👤 Get User Info                       |
| `/api/v1/predict`              | POST   | 🔍 Predict License Plate & Vehicle Type |
| `/api/v1/uploads/<filename>`   | GET    | 🖼 Retrieve Uploaded Image             |

---

## 📥 Example Request
To predict license plates and vehicle types:
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Authorization: Bearer <your_jwt_token>" \
  -F "file=@/path/to/your/image.jpg"
```
---

## 📈 System Design

```mermaid
flowchart LR
    A[Data] --> B(Training)
    B --> C[API]
    C --> D[UI]
    D --> E((Prédiction))

    %% Annotations
    click A href "#data-preparation" _blank
    click B href "#model-training" _blank
```

```mermaid
flowchart TD
 subgraph subGraph0["Détails des étapes"]
        A1[("License Plate Recognition")]
        n1@{ label: "<span style=\"padding-left:\">Vehicle Images Dataset</span>" }
        A["Data Preparation"]
        B1{{"Models"}}
        B["Model Training"]
        C1["/predict (Flask)"]
        C["API Deployment"]
        D1["Streamlit App"]
        D["User Interface"]
  end
    A --> B & A1 & n1
    B --> C & B1
    C --> D & C1
    D --> E["Prediction"] & D1

    n1@{ shape: cylinder}
```

```mermaid
sequenceDiagram
    participant User as User
    participant API as API (Flask Server)
    participant Processing as Preprocessing
    participant YOLO as YOLOv8 Model
    participant Classifier as Vehicle Classifier

    User ->>+ API: POST /predict (Upload Image)
    API ->> Processing: Resize, Normalize, Preprocess
    Processing ->> YOLO: Detect License Plate
    YOLO -->> API: Bounding Box + Cropped Plate
    Processing ->> Classifier: Classify Vehicle Type
    Classifier -->> API: Vehicle Type Prediction
    API -->>- User: JSON Response (Plate + Vehicle Type)
```


### 🛠 Detailed Steps

- **Business Understanding**  
  ➔ Build an intelligent system to detect license plates and classify vehicle types.

- **Data Understanding**  
  ➔ Collected datasets for license plates (YOLO format) and vehicle classification (images labeled by type).

- **Data Preparation**  
  ➔ Preprocessing images, formatting datasets for YOLOv8 and TensorFlow.

- **Modeling**  
  ➔ Trained YOLOv8 for license plate detection.  
  ➔ Built and trained a TensorFlow model for vehicle type classification.

- **Deployment**  
  ➔ Developed a secure Flask backend serving predictions via REST API.  
  ➔ Developed a Streamlit web app

---

## 📚 Datasets

### License Plate Dataset
- **Source**: Collected and pre-annotated via [Roboflow](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e).
- **Format**: YOLOv8 bounding box annotations.
- **Contents**: Diverse license plates under different lighting and angles.

- <img src="https://stage-drupal.car.co.uk/s3fs-public/styles/original_size/public/2019-09/why-are-number-plates-yellow-and-white.jpg?rt1UJUyIi7L2DpS613hFYlI5ng3U4QT3&itok=3SZjXU0B" width="200px"/>

---

### Vehicle Type Classification Dataset
- **Source**: Dataset from [Kaggle - Vehicle Images Dataset](https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset).
- **Categories**:  'Big Truck', 'City Car', 'Multi Purpose Vehicle','Sedan', 'Sport Utility Vehicle', 'Truck' and 'Van'

- <img src="https://storage.googleapis.com/kagglesdsdata/datasets/1419951/2351806/Big%20Truck/Image_000001.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250427%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250427T061120Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=724c9413a0dec5cd77e4ced6c881504c926c57215a33ffdfa29a3cb32e0e50899ea80282f2d32b9fff35318685fe500d4258c9934b540fc1891c6298ffa0e58bb93cd89c80c1026ad6c3e7810cf957ecef3fa2c7891be13a07f9212fbfcf290328f81f71a82c042be72827a4d8353836a9f35605cc5c03f1010c304150115a1241f6587bb1ee1f62bfd196e49ff11b4d841afda88060228de5887600cfb7e01be3f01685e1817506d5f87fc4a013b75e16a356651ea98b4cef7c4489bb173e35410b3eacad7d901037b989e5104affcc98fd84c46268c7328ab6a14b9339376d0b89e7a2266a3930ff30173e9ae89bab2ef4b3d4dcbfd2487e76847b95c9506a" width="200px"/>

---

## 📊 Model Performance and Statistics

### License Plate Detection - YOLOv8
- **Model**: YOLOv8
- **Purpose**: Detect and localize license plates on vehicle images.
- **Performance**: Achieved a high **mean Average Precision (mAP)**, ensuring robust detection even under varying lighting conditions and angles.
- **Highlights**:
  - Real-time detection capability.
  - Lightweight and efficient, perfect for production-level deployment.

### Vehicle Type Classification - MobileNetV2 + Custom CNN
- **Model**: Custom CNN model based on **MobileNetV2**.
- **Purpose**: Classify vehicle types (e.g., Car, Truck, Bus).
- **Training**:
  - Fine-tuned on the collected dataset.
  - Applied data augmentation for better generalization.

- **Final Accuracy**: **69.8%**

### Training Graphs

**Accuracy over epochs**:

<img src="training_history/accuracy.png" width="400px"/>

**Loss over epochs**:

<img src="training_history/loss.png" width="400px"/>

---


## 👥 Contributors
**TODO**

