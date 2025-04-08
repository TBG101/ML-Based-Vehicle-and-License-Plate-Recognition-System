from roboflow import Roboflow
from ultralytics import YOLO
import os
import dotenv


def download_plate_dataset():
    dotenv.load_dotenv()
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(
        "roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    version = project.version(11)
    dataset = version.download("yolov8")
    return dataset.location


def train_model():
    model = YOLO("yolov8n.pt")

    # Download the dataset
    dataset_location = download_plate_dataset()

    results = model.train(data=os.path.join(
        dataset_location, "data.yaml"), epochs=10)

    # Save the trained model
    model.save("models/plate_recognition_yolo_model.pt")
