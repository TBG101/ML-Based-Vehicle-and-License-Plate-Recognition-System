import streamlit as st
import requests

# Update this to your backend URL if needed
BACKEND_URL = "http://localhost:5000"

# Title
st.title("Login to Upload Your Car Image")


# Initialize session state
if "token" not in st.session_state:
    st.session_state.token = None

# Login form
if st.session_state.token is None:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            try:
                response = requests.post(
                    BACKEND_URL+"/api/v1/login",
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
    st.header("Upload Your Car Image")

    uploaded_file = st.file_uploader(
        "Choose a car image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Car Image",
                 use_container_width=True)

        if st.button("Send to Predict API"):
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
                    BACKEND_URL+"/api/v1/predict",
                    files=files,
                    headers=headers
                )

                if response.status_code == 200:
                    # Display the results
                    prediction = response.json()
                    st.success("Prediction received!")
                    st.write(
                        f"**Number Plate(s):** {prediction['number_plate']}")
                    st.write(f"**Car Type:** {prediction['car_type']}")
                    st.image(
                        f"{BACKEND_URL}/api/v1/{prediction['image_url']}", caption="Processed Image", use_container_width=True)
                else:
                    st.error(f"Prediction failed: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error sending to predict API: {e}")
