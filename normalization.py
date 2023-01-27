import numpy as np
import cv2
import albumentations as A
import glob

epi_dirs = glob.glob('./make_bacteria_map/epidermidis_crop/*.png')
aur_dirs = glob.glob('./make_bacteria_map/aureus_crop/*.png')


epi_imgs = []
aur_imgs = []


def find_mean_std(img_lsts):

    meanRGB = [np.mean(x, axis=(0,1)) for x in img_lsts]
    stdRGB = [np.std(x, axis=(0,1)) for x in img_lsts]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])

    return((meanR, meanG, meanB), (stdR, stdG, stdB))

for i in epi_dirs:
    img = cv2.imread(i, cv2.IMREAD_COLOR) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    epi_imgs.append(img)

for i in aur_dirs:
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aur_imgs.append(img)
    
print(find_mean_std(epi_imgs))
print(find_mean_std(aur_imgs))





