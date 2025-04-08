import kagglehub


def download_data():
    """
    Downloads the vehicle images dataset from Kaggle and copies it to the current working directory.
    """

    # Download the dataset from Kaggle
    path = kagglehub.dataset_download("lyensoetanto/vehicle-images-dataset")

    if path is None:
        print("Failed to download the dataset.")
        return

    return path
