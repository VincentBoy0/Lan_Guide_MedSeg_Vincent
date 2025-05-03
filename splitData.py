import os
import pandas as pd
import re

def extract_id(filename):
    """Extract number from filename for matching, e.g. img_001.png -> 001"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def generate_csv(images_dir, labels_dir, output_csv):
    image_files = os.listdir(images_dir)
    label_files = os.listdir(labels_dir)

    image_dict = {extract_id(f): f for f in image_files}
    label_dict = {extract_id(f): f for f in label_files}

    common_ids = set(image_dict.keys()) & set(label_dict.keys())

    data = []
    for cid in sorted(common_ids):
        data.append({
            'Image': os.path.join(images_dir, image_dict[cid]),
            'Description': os.path.join(labels_dir, label_dict[cid])
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} with {len(df)} entries")

# Paths
train_img_dir = './data/QaTa-COV19-v2/Train/Images'
train_label_dir = './data/QaTa-COV19-v2/Train/Ground-truths'
test_img_dir = './data/QaTa-COV19-v2/Test/Images'
test_label_dir = './data/QaTa-COV19-v2/Test/Ground-truths'

# Output
os.makedirs('./data/QaTa-COV19-v2/prompt', exist_ok=True)
generate_csv(train_img_dir, train_label_dir, './data/QaTa-COV19-v2/prompt/train.csv')
generate_csv(test_img_dir, test_label_dir, './data/QaTa-COV19-v2/prompt/test.csv')
