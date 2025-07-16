# %% [markdown]
# Importujeme všechny potřebné knihovny
# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# %% [markdown]
# Načteme obrázek
root = os.getcwd()
img_path = os.path.join(root, 'lesson02/assignment02/pencils.png')
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# %%
# 1) Výpočet převodu do černobílé podle vzorců z prezentace
# Převod z sRGB
img2 = img.astype(np.float32) / 255
img_line = np.where(img2 <= 0.04045, img2 / 12.92, ((img2 + 0.055) / 1.055) ** 2.4)

lumi = 0.2127 * img_line[:,:,2] + 0.7152 * img_line[:,:,1] + 0.0722 * img_line[:,:,0]

# %%
# Převod zpět do sRGB
srgb = np.where(lumi <= 0.0031308, lumi * 12.92, 1.055 * np.power(lumi, 1/2.4) - 0.055) * 255

gray = (srgb).astype(np.uint8)

# %%
# Přepočet z relativní luminance 
b, g, r = img[...,0], img[...,1], img[...,2]
gray_match = (0.11 * b + 0.59 * g + 0.3 * r).round().clip(0,255).astype(np.uint8)

plt.imshow(gray_match, 'gray')
plt.title("Černobílá vypočítaná")

# %%
# OpenCV černobílá
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_cv, 'gray')
plt.title("Černobílá z OpenCV")
