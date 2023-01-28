import glob
import cv2
import argparse

parser = argparse.ArgumentParser(description='bacteria_map')


# 입력받을 인자값 설정 (default 값 설정가능)
parser.add_argument('--epi_dir', type=str, default='./epidermidis_crop/*.png')
parser.add_argument('--aur_dir', type=str, default='./aureus_crop/*.png')
parser.add_argument('--cellcount', type=tuple, default=(10, 20), help='input 2 number (min ,  max) e.g. 8, 11')
parser.add_argument('--map_count', type=int, default=300, help='how many maps you generate')
# args 에 위의 내용 저장
args = parser.parse_args()

aur_dirs = glob.glob('./S. aureus_full/*.png')
epi_dirs = glob.glob('./S. epidermidis_full/*.png')

def read(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (2048, 2048))
    img1 = img[0:1024, 0:1024, :]
    img2 = img[1024:2048, 0:1024, :]
    img3 = img[0:1024, 1024:2048, :]
    img4 = img[1024:2048, 1024:2048, :]
    
    
    cv2.imwrite('./epidermidis_full_1024/'+ img_dir.split('/')[-1].split('.')[0] + '_1.png', img1)
    cv2.imwrite('./epidermidis_full_1024/'+ img_dir.split('/')[-1].split('.')[0] + '_2.png', img2)
    cv2.imwrite('./epidermidis_full_1024/'+ img_dir.split('/')[-1].split('.')[0] + '_3.png', img3)
    cv2.imwrite('./epidermidis_full_1024/'+ img_dir.split('/')[-1].split('.')[0] + '_4.png', img4)
    
for i in epi_dirs:
    read(i)

