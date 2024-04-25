import streamlit as st
from image_utils import process_images, cluster_images
from web_scraper import fetch_images

def main():
    st.title('Image Clusters')

    url = st.text_input('Enter a URL to scrape images:')
    if url:
        images = fetch_images(url)  # This should return a list of tuples (img_url, description)
        if images:
            # Extract image URLs; each image is a tuple (url, description)
            image_urls = [img[0] for img in images]
            features = process_images(image_urls)
            labels = cluster_images(features)

            # Create a dictionary to hold clusters of images
            clustered_images = {label: [] for label in set(labels)}
            for label, image_data in zip(labels, images):
                clustered_images[label].append(image_data)

            # Display each cluster and its images
            for cluster_id, imgs in clustered_images.items():
                st.subheader(f"Cluster {cluster_id}")
                for img_url, description in imgs:
                    # Specify image width to reduce display size, Streamlit adjusts height automatically
                    st.image(img_url, caption=description, width=300)  # You can adjust the width as needed
        else:
            st.write("No images found at the URL.")

if __name__ == '__main__':
    main()
