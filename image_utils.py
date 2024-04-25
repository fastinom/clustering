from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import base64

# Load the VGG16 model with weights pre-trained on ImageNet.
model = VGG16(weights='imagenet', include_top=False)

def process_image(img_url):
    """Process an individual image from a URL or a data URL."""
    try:
        if img_url.startswith('data:image'):  # Check if img_url is a data URL
            # Extract and decode the base64 part of the data URL.
            base64_str = img_url.split('base64,')[-1]
            image_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(image_data))
        else:
            # Fetch the image over HTTP/HTTPS.
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
        
        # Resize image to 224x224 for VGG16 input compatibility.
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # Predict to extract features.
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image {img_url}: {str(e)}")
        return None  # Return None if there's an error processing the image.

def process_images(urls):
    """Process a list of image URLs into a list of feature arrays."""
    features = [process_image(url) for url in urls]
    # Filter out None values if any images failed to process.
    return [feature for feature in features if feature is not None]

def cluster_images(features, n_clusters=5):
    """Cluster feature arrays into n clusters."""
    if not features:
        return []
    try:
        # Convert the list of features into a numpy array for clustering.
        feature_array = np.array(features)
        # Check if feature_array is correctly formed.
        if len(feature_array.shape) != 2:
            raise ValueError("Feature array shape is incorrect. Check feature extraction.")
        # Perform k-means clustering.
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(feature_array)
        return labels
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        return []
