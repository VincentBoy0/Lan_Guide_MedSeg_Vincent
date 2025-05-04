import os
import pandas as pd

# Define paths
root_path = "./data/QaTa-COV19-v2/Test"
image_dir = "Images"

# Get list of image files
image_files = os.listdir(os.path.join(root_path, image_dir))

# Define captions based on file names or patterns
captions = []
for img in image_files:
    if "normal" in img.lower():
        captions.append("A normal chest X-ray with no visible abnormalities.")
    elif "pneumonia" in img.lower():
        captions.append("A chest X-ray showing signs of pneumonia.")
    elif "covid" in img.lower():
        captions.append("A chest X-ray showing signs of COVID-19 infection.")
    else:
        captions.append("A chest X-ray with evidence of lung infection.")

# Ensure the number of captions matches the number of images
if len(image_files) != len(captions):
    raise ValueError("Number of images and captions must match.")

# Create a DataFrame
data = {
    "Image": [os.path.join(image_dir, img) for img in image_files],
    "Description": captions
}
df = pd.DataFrame(data)

# Save to CSV
csv_path = "./data/QaTa-COV19-v2/prompt/test.csv"
print(csv_path)
df.to_csv(csv_path, index=False)
print(f"train.csv saved to {csv_path}")