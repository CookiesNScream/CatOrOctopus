import kagglehub
import tensorflow as tf

def download_data():
    # Download latest version
    path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
    print("Path to dataset files:", path)

    path = kagglehub.dataset_download("crawford/cat-dataset")
    print("Path to dataset files:", path)

    path = kagglehub.dataset_download("vencerlanz09/sea-animals-image-dataste")
    print("Path to dataset files:", path)