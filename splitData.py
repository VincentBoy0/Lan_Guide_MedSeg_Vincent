import os
import pandas as pd

# Define paths
root_path = "./data/QaTa-COV19-v2/prompt/train_1.txt"

image_list = []
caption_list = []

with open(root_path, "r") as file:
    current_line = ""
    for line in file:
        # Check if the line starts with an image path (assumes paths don't contain spaces)
        if line.startswith("mask_"):
            # Process the previous line if it exists
            if current_line:
                path, description = current_line.split('\t', 1)
                image_list.append(path.strip())
                caption_list.append(description.strip())
            # Start a new line
            current_line = line.strip()
        else:
            # Append the continuation of the description
            current_line += " " + line.strip()

    # Process the last line
    if current_line:
        path, description = current_line.split('\t', 1)
        image_list.append(path.strip())
        caption_list.append(description.strip())

# Create a DataFrame
data = {
    "Image": image_list,
    "Description": caption_list
}
df = pd.DataFrame(data)

# Save to CSV
csv_path = "./data/QaTa-COV19-v2/prompt/train.csv"
print(csv_path)
df.to_csv(csv_path, index=False)
print(f"train.csv saved to {csv_path}")