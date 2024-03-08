from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import warnings
warnings.filterwarnings("ignore")

model = create_model("Unet_2020-10-30")
model.eval()
image = load_rgb("./static/cloth_web.jpg")

transform = albu.Compose([albu.Normalize(p=1)], p=1)

padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

with torch.no_grad():
    prediction = model(x)[0][0]

mask = (prediction > 0).cpu().numpy().astype(np.uint8)
mask = unpad(mask, pads)

img = np.full((1024, 768, 3), 255)
seg_img = np.full((1024, 768), 0)

b = cv2.imread("./static/cloth_web.jpg")
b_img = mask * 255

# Calculate the exact dimensions that will fit into the img array
# Ensure both dimensions are even to avoid mismatch
exact_height = 1024 - (1024 % 2)
exact_width = 768 - (768 % 2)

# Resize b and b_img to these exact dimensions
b = cv2.resize(b, (exact_width, exact_height))
b_img = cv2.resize(b_img, (exact_width, exact_height))

# Now, the dimensions of b and b_img should match the slice of img
# You can proceed with the assignment without encountering the ValueError
img[int((1024-exact_height)/2): 1024-int((1024-exact_height)/2), int((768-exact_width)/2):768-int((768-exact_width)/2)] = b
seg_img[int((1024-exact_height)/2): 1024-int((1024-exact_height)/2), int((768-exact_width)/2):768-int((768-exact_width)/2)] = b_img

# Save the images
cv2.imwrite("./HR-VITON-main/test/test/cloth/00001_00.jpg", img)
cv2.imwrite("./HR-VITON-main/test/test/cloth-mask/00001_00.jpg", seg_img)
