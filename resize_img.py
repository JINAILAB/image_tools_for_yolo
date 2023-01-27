import glob
import cv2

epi_dirs = glob.glob('./make_bacteria_map/epidermidis_crop/*.png')
aur_dirs = glob.glob('./make_bacteria_map/aureus_crop/*.png')

for i in aur_dirs:
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (60,60))
    cv2.imwrite('aur_60/'+i.split('/')[-1], img)
    
for i in epi_dirs:
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (60,60))
    cv2.imwrite('epi_60/'+i.split('/')[-1], img)
    