import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.dataset import QaTa
import utils.config as config
from engine.wrapper import LanGuideMedSegWrapper
import torchvision.transforms.functional as TF
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/training.yaml', type=str)
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def visualize_sample(image, gt, pred, index):
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))

  # Ensure the image is in the correct orientation
  axs[0].imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert to (H, W, C)
  axs[0].set_title("Original Image")

  axs[1].imshow(gt.squeeze().cpu().numpy(), cmap='gray')  # Ground truth
  axs[1].set_title("Ground Truth Mask")

  axs[2].imshow(pred.squeeze().cpu().numpy(), cmap='gray')  # Predicted mask
  axs[2].set_title("Predicted Mask")

  for ax in axs:
      ax.axis('off')

  # Save the figure
  output_dir = './output_images/'
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  output_path = os.path.join(output_dir, f'image_{index}.png')
  plt.savefig(output_path)

  # Show the image in the notebook (important for Colab)
  plt.show()

  plt.close()  # Close the figure to free memory
if __name__ == '__main__':
    args = get_parser()

    # Load model
    model = LanGuideMedSegWrapper(args)
    #checkpoint = torch.load('./save_model/medseg.ckpt', map_location='cuda')
    checkpoint = torch.load('./save_model/medseg.ckpt', map_location='cuda', weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval().cuda()

    # Load test dataset
    ds_test = QaTa(csv_path=args.test_csv_path,
                   root_path=args.test_root_path,
                   tokenizer=args.bert_type,
                   image_size=args.image_size,
                   mode='test')

    dl_test = DataLoader(ds_test, batch_size=1, shuffle=True)

    # Visualize predictions for a few samples
    with torch.no_grad():
        for i, ([img, text], gt) in enumerate(dl_test):
            if i >= 5:  # show 5 samples
                break

            img = img.cuda()
            for key in text:
                text[key] = text[key].cuda()

            pred = model((img, text))
            pred_mask = torch.sigmoid(pred) > 0.7  # Binarize output
            #print("Raw Predictions:", pred)
            #print("Sigmoid Predictions:", torch.sigmoid(pred))
            visualize_sample(img[0], gt[0], pred_mask[0], i)
