import numpy as np
import cv2
import glob
import random
import argparse

# 인자값을 받을 수 있는 인스턴스 생성

parser = argparse.ArgumentParser(description='bacteria_map')

# 입력받을 인자값 설정 (default 값 설정가능)
parser.add_argument('--epi_dir', type=str, default='./epidermidis_crop/*.png')
parser.add_argument('--aur_dir', type=str, default='./aureus_crop/*.png')
parser.add_argument('--cellcount', type=tuple, default=(10, 20), help='input 2 number (min ,  max) e.g. 8, 11')
parser.add_argument('--map_count', type=int, default=300, help='how many maps you generate')
# args 에 위의 내용 저장
args = parser.parse_args()


# 입력받은 인자값 출력
epi_ls = glob.glob(args.epi_dir)
aur_ls = glob.glob(args.aur_dir)
print(len(aur_ls))
print(len(epi_ls))




# cell 붙이기전에 resize 해주는 과정 (h,w) : (50, 50) ~ (80, 80)
def make_cell(img_name):
    rn = random.uniform(0, 1) * 0.3
    color_img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    color_img = cv2.resize(color_img, (int(rn * 100) + 50, int(rn * 100) + 50))
    return color_img


# 이미지 파일 받아서 background에 랜덤으로 붙여주는 과정
# yolov5 cls x0 y0 w h
# black ground

def make_map(file_name, background):
    if file_name.split('_')[0] =='./epidermidis':
        print('epi')
        cls_num = 1
    elif file_name.split('_')[0] == './aureus':
        cls_num = 0
        print('aur')
    
    
    cell_img = make_cell(file_name)
    img_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    ret, img_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

    background_h = background.shape[0]
    background_w = background.shape[1]

    img_mask_inv = cv2.bitwise_not(img_mask)
    height, width = cell_img.shape[:2]

    # cell_img의 height와 width 갖고오기

    # 아무곳에나 붙이기
    ran_num_h = random.randint(3, background_h - 100)
    ran_num_w = random.randint(3, background_w - 100)
    img_roi = background[ran_num_h:ran_num_h + height, ran_num_w:ran_num_w + width]




    x0 = (ran_num_w + width / 2) / background_w
    y0 = (ran_num_h + height / 2) / background_h
    w = width / background_w
    h = height / background_h

    img1 = cv2.bitwise_and(cell_img, cell_img, mask=img_mask_inv)

    # 배경인 부분을 추출해줌. 나머지는 0
    img2 = cv2.bitwise_and(img_roi, img_roi, mask=img_mask_inv)

    dst = cv2.add(img1, img2)

    background[ran_num_h:ran_num_h + height, ran_num_w:ran_num_w + width] = dst

    return background, [cls_num, x0, y0, w, h]




# print(len(epi_ls)) 433
# print(len(aur_ls)) 275

background = np.zeros((1024, 1024, 3), dtype=np.uint8)


for j in range(args.map_count):
    background = np.zeros((1024, 1024, 3), dtype=np.uint8)
    cell_num = random.randint(*args.cellcount)
    for i in range(cell_num):
        aur_order = random.randint(0, len(aur_ls)-1)
        epi_order = random.randint(0, len(epi_ls)-1 )
        background, cls_info1 = make_map(aur_ls[aur_order], background)
        background, cls_info2 = make_map(epi_ls[epi_order], background)
        with open(f'./bacteria_map/{j}.txt', 'a') as f:
            f.write(' '.join((str(s) for s in cls_info1))+'\n')
            f.write(' '.join((str(s) for s in cls_info2))+'\n')
            

    cv2.imwrite(f'./bacteria_map/{j}.png', background)