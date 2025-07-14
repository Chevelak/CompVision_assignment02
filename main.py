# %% [markdown]
# Importujeme všechny potřebné knihovny
# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
import os

# %% [markdown]
# Načteme obrázek
root = os.getcwd()
img_path = os.path.join(root, 'lesson02/assignment02/pencils.png')
img = cv2.imread(img_path)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.show()

plt.figure()
plt.imshow(imgRGB)
plt.show()

# %%
# Zadané obrázky na převedení
# %%
lab_cv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[...,0]
lab_sk = rgb2lab(img[..., [2,1,0]])[..., 0]
lab_my = img.sum(-1) # Doplňte správný výpočet

# %%
plt.figure(figsize=(12,3))
plt.subplot(1, 3, 1)
plt.imshow(lab_cv)
plt.title("Luminance z OpenCV")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(lab_sk)
plt.title("Luminance z skimage")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(lab_my)
plt.title("Vypočítaná hodnota luminance")
plt.axis("off")


# %%
# Z luminiscence do černobílé
img2 = cv2.imread(img_path)
gray_cv = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
graysk = rgb2lab(img2[..., [0,1,2]])[..., 0]
gray_sk = (graysk.astype(np.uint8))*255
gray_my = (
    0.2 * img2[..., 0] +
    0.5 * img2[..., 1] +
    0.3 * img2[..., 2]
)

# %%
plt.figure(figsize=(12,3))
plt.subplot(1, 3, 1)
plt.imshow(gray_cv, 'gray')
plt.title("Černobílá z OpenCV")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(gray_sk, 'gray')
plt.title("Černobílá z skimage")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(gray_my, 'gray')
plt.title("Vypočítaná hodnota černobílá")
plt.axis("off")



# %%
# Vrácení luminiscence zpět do normálu
cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sk = img[..., [2,1,0]]
my = (
    np.dstack((
    (img[...,2] - img[...,2].min()) * (255/(img[...,2].max() - img[...,2].min())),
    (img[...,1] - img[...,1].min()) * (255/(img[...,1].max() - img[...,1].min())),
    (img[...,0] - img[...,0].min()) * (255/(img[...,0].max() - img[...,0].min()))
    ))
    .astype(np.uint8)
)

# %%
plt.figure(figsize=(12,3))
plt.subplot(1, 3, 1)
plt.imshow(cv)
plt.title("Zpět z lumi z OpenCV")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(sk)
plt.title("Zpět z lumi z skimage")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(my)
plt.title("Vypočítaná hodnota zpět z lumi")
plt.axis("off")
