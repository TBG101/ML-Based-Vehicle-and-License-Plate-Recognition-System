import streamlit as st
import requests

# Update this to your backend URL if needed
BACKEND_URL = "http://localhost:5000"

# Title
st.title("Login to Upload Your Car Image")

# Initialize session state
if "token" not in st.session_state:
    st.session_state.token = None

# Check if the user is logged in
if st.session_state.token is None:
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            with st.spinner("Logging in..."):
                try:
                    response = requests.post(
                        BACKEND_URL + "/api/v1/login",
                        json={"username": username, "password": password},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        token = data.get("token")
                        if token:
                            st.session_state.token = token
                            st.success("Login successful!")
                        else:
                            st.error("Token not found in response.")
                    else:
                        st.error(f"Login failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API: {e}")

else:
    # If logged in, show the upload image section
    st.header("Upload Your Car Image")

    uploaded_file = st.file_uploader(
        "Choose a car image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Car Image",
                 use_container_width=True)

        if st.button("Send to Predict API"):
            with st.spinner("Sending image for prediction..."):
                try:
                    # Prepare the file to be sent to the API
                    files = {
                        "image": (uploaded_file.name, uploaded_file, uploaded_file.type)
                    }
                    headers = {
                        "Authorization": f"Bearer {st.session_state.token}"
                    }

                    # Send POST request to the predict endpoint
                    response = requests.post(
                        BACKEND_URL + "/api/v1/predict",
                        files=files,
                        headers=headers
                    )

                    if response.status_code == 200:
                        # Display the results
                        prediction = response.json()
                        st.success("Prediction received!")
                        car_type = prediction.get("car_type", "Unknown")
                        car_type_confidence = car_type.split(
                        )[-1] if car_type else "Unknown"
                        car_type = " ".join(car_type.split(" ")[:-1]).strip()
                        st.write(f"**Car Type:** {car_type}")
                        st.write(
                            f"**Car Type Confidence:** {car_type_confidence}")

                        if prediction['number_plate'] is not None and len(prediction['number_plate']) > 0:
                            st.write("**License Plate Images:**")

                        for plate in prediction['number_plate']:
                            try:
                                # Try to display each license plate image
                                plate_url = f"{BACKEND_URL}/api/v1/{plate}"
                                st.image(
                                    plate_url, caption=f"License Plate: {plate}", use_container_width=True)
                            except Exception as e:
                                st.error(
                                    f"Error displaying license plate image for {plate}: {e}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error sending to predict API: {e}")
