import cv2
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser(description='이미지에서 중앙에 있는 하나의 cell만 예쁘게 잘라줍니다.')

parser.add_argument('--classes', type =str, default='aureus', help = "can use 'aureus' or 'epidermidis'")

args = parser.parse_args()

if args.classes == 'aureus':
    args.image_dir = glob.glob('./S. aureus_crop/*.png')
elif args.classes == 'epidermidis':
    args.image_dir = glob.glob('./S. epidermidis_crop/*.png')



def find_center_circle(img_str : str) -> np.ndarray:
    
    img_clr = cv2.imread(img_str, cv2.IMREAD_COLOR)
    img_clr = img_clr[100:1948, 100:1948]
    img_gray = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.resize(img_gray, dsize=(2048, 2048))
    img_gray = img_gray[100:1948, 100:1948]

    ret, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    img_result = cv2.medianBlur(img_thresh, 21)
    ret, img = cv2.threshold(img_result, 40, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # black = np.zeros((2048, 2048, 3), dtype=np.uint8)
    for find_img in range(len(contours)):
        # num_list = list(range(1, 100, 10))
        # num_list.append(list(range(1945, 2048, 10)))

        ctrs = contours[find_img]
        ctrs = ctrs.reshape(-1)
        img_srts = None

        if 0 not in ctrs and 20 not in ctrs and 30 not in ctrs and 60 not in \
                ctrs and 90 not in ctrs and 1947 not in ctrs and 1928 not in ctrs and 1890 not in \
                ctrs and 1910 not in ctrs and 100 not in ctrs and 1920 not in ctrs and 1880 not in \
                ctrs:
            x, y, w, h = cv2.boundingRect(contours[find_img])

            # cv2.drawContours(img_clr, contours, find_img, (0, 255, 0), 60)

            if w > 700 and h > 700:
                # img_clr = cv2.rectangle(img_clr, (x, y), (x + w, y + h), (255, 100, 100), 60)
                img_srts = img_clr[y - 50: y + h + 50, x - 50: x + w + 50]

        else:
            pass

    return img_srts


for img in args.image_dir:
    img_color = cv2.imread(img, cv2.IMREAD_COLOR)
    img_color = cv2.resize(img_color, dsize=(2048, 2048))

    img_name = img.split('/')[-1]
    #S. 을 뗀 파일에 저장 
    filename = img.split('/')[-2].split('.')[-1].split()[0]

    a = find_center_circle(img)

    try:
        cv2.imwrite(f'./{filename}/{img_name}', a)
    except:
        pass