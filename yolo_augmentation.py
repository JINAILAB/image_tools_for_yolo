import albumentations as A
import cv2
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='augmentation_for_yolo')

# 입력받을 인자값 설정 (default 값 설정가능)
parser.add_argument('--folder_dir', type=str, default='data')
parser.add_argument('--preimg_resize', type=lambda x: tuple(map(int, x.split(','))), default='4096, 4096')
parser.add_argument('--postimg_resize', type=lambda x: tuple(map(int, x.split(','))), default='1024, 1024',
                    help='The desired output of the final image')
parser.add_argument('--crop_division_num', type=int, default=4, help='How many levels will you divide the image?')
# args 에 위의 내용 저장
args = parser.parse_args()

cwd = os.getcwd()


class Transforms:
    def __init__(self, folder_dir, preimg_resize, postimg_size, crop_division_num):
        self.cwd = os.getcwd()
        self.transforms_dict = {}
        self.dir = folder_dir
        self.preimg_resize = preimg_resize
        self.postimg_size = postimg_size
        self.crop_division_num = crop_division_num

    def get_img(self, img_dir):
        img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.preimg_resize)
        return img

    def get_label(self, labels_dir):
        bboxes = []
        classes = []
        with open(labels_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                bboxes.append(list(map(float, line.split(' ')))[1:])
                classes.append(line.split(' ')[0])
        return bboxes, classes

    def write_file(file_dir, img=None, boxes=None, classes=None):
        img_write_dir = './images/' + file_dir + '.png'
        txt_write_dir = './labels/' + file_dir + '.txt'
        cv2.imwrite(img_write_dir, img)
        with open(txt_write_dir, 'w') as f:
            for i in range(len(boxes)):
                strs = classes[i] + ' ' + ' '.join(str(s) for s in boxes[i]) + '\n'
                f.write(strs)

    def img_crop(self, img):
        images = []
        assert img is not None, 'image is empty'
        cropimg_size = img.shape[0] // self.crop_division_num
        for i in range(self.crop_division_num):
            images.append(img[i*cropimg_size:(i+1)*cropimg_size, i*cropimg_size:(i+1)*cropimg_size])

        return images

    def img_label_crop(self, img, bboxes=None, classes=None):
        images = []
        labels = []
        assert img is not None, 'image is empty'
        assert bboxes is not None, 'bboxes is empty'
        assert classes is not None, 'classes is empty'

        cropimg_size = img.shape[0] // self.crop_division_num

        for i in range(0, self.crop_division_num):
            transform = A.Compose([
                A.Crop(x_min=i * cropimg_size, y_min=i * cropimg_size, x_max=(i + 1) * cropimg_size,
                       y_max=(i + 1) * cropimg_size)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            transformed = transform(image=img, bboxes=bboxes, class_labels=classes)

            img_tr = transformed['image']
            bboxes_tr = transformed['bboxes']
            classes_tr = transformed['class_labels']

            img_tr = cv2.resize(img_tr, self.postimg_size)

            images.append(img_tr)

            label = []
            for j in zip(classes_tr, bboxes_tr):
                print(type(j[0]))
                ls = []
                label.append([int(j[0]), *j[1]])
            labels.append(label)
        return images, labels








if __name__ == '__main__':

    a = Transforms(args.folder_dir, args.preimg_resize, args.postimg_resize, args.crop_division_num)
    img = a.get_img('/Users/jinyong/Desktop/tools for detection/data/images/0818 s2.1.png')
    labels = a.get_label('/Users/jinyong/Desktop/tools for detection/data/labels/aur_0_1.txt')
    x, y = a.img_label_crop(img, *labels)
    print(a.img_label_crop(img, *labels))
    print(len(x), len(y))

# transform1 = A.Compose([
#     A.Crop(x_min=0, y_min=0, x_max=1024, y_max=1024)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# transform2 = A.Compose([
#     A.Crop(x_min=1024, y_min=0, x_max=2048, y_max=1024)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# transform3 = A.Compose([
#     A.Crop(x_min=0, y_min=1024, x_max=1024, y_max=2048)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# transform4 = A.Compose([
#     A.Crop(x_min=1024, y_min=1024, x_max=2048, y_max=2048)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# bacteria = { '0' : 'aureus', '1' : 'epimidis'}
# img_dirs = glob.glob('./2048_images/*.png')
# img_dirs.sort()
# txt_dirs = glob.glob('./2048_labels/*.txt')
# txt_dirs.sort()


# def getfile(img_file_dir, txt_file_dir):
#     img = cv2.imread(img_file_dir, cv2.IMREAD_COLOR)
#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     bboxes = []
#     classes = []
#     with open(txt_file_dir, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             line = line.strip()
#             bboxes.append(list(map(float, line.split(' ')))[1:])
#             classes.append(line.split(' ')[0])
#     print(img.shape)
#     return img, bboxes, classes


# for i in range(len(img_dirs)):
#     img, bboxes, classes = getfile(img_dirs[i], txt_dirs[i])
#     transformed1 = transform1(image=img, bboxes=bboxes, class_labels=classes)
#     transformed2 = transform2(image=img, bboxes=bboxes, class_labels=classes)
#     transformed3 = transform3(image=img, bboxes=bboxes, class_labels=classes)
#     transformed4 = transform4(image=img, bboxes=bboxes, class_labels=classes)

#     dir1 = img_dirs[i].split('.')[-2].split('/')[-1] +'_1'
#     dir2 = img_dirs[i].split('.')[-2].split('/')[-1] +'_2'
#     dir3 = img_dirs[i].split('.')[-2].split('/')[-1] +'_3'
#     dir4 = img_dirs[i].split('.')[-2].split('/')[-1] +'_4'

#     img_tr1 = transformed1['image']
#     bboxes_tr1 = transformed1['bboxes']
#     classes_tr1 = transformed1['class_labels']

#     img_tr2 = transformed2['image']
#     bboxes_tr2 = transformed2['bboxes']
#     classes_tr2 = transformed2['class_labels']

#     img_tr3 = transformed3['image']
#     bboxes_tr3 = transformed3['bboxes']
#     classes_tr3 = transformed3['class_labels']

#     img_tr4 = transformed4['image']
#     bboxes_tr4 = transformed4['bboxes']
#     classes_tr4 = transformed4['class_labels']

#     write_file(dir1, img_tr1, bboxes_tr1, classes_tr1)
#     write_file(dir2, img_tr2, bboxes_tr2, classes_tr2)
#     write_file(dir3, img_tr3, bboxes_tr3, classes_tr3)
#     write_file(dir4, img_tr4, bboxes_tr4, classes_tr4)
