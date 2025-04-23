# preload.py

print("🧠 Model preload initializing...")

model_ready = False
get_license_plates = None
carPredict = None

def preload_models():
    global model_ready, get_license_plates, carPredict
    try:
        from plateRecognition.plate_recognition import get_license_plates as plate_fn
        from carRecognition.car_recogntion import predict as car_fn

        # Assign them to globals so your app can use them
        get_license_plates = plate_fn
        carPredict = car_fn

        model_ready = True
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
