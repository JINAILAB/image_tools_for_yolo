import albumentations as A
import cv2
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='augmentation_for_yolo')

# 입력받을 인자값 설정 (default 값 설정가능)
parser.add_argument('--folder_dir', type=str, default='data')
parser.add_argument('--preimg_resize', type=lambda x: tuple(map(int, x.split(','))), default='4096, 4096')
parser.add_argument('--postimg_resize', type=lambda x: tuple(map(int, x.split(','))), default='1024, 1024',
                    help='The desired output of the final image')
parser.add_argument('--crop_division_num', type=int, default=4, help='How many levels will you divide the image?')
# args 에 위의 내용 저장
args = parser.parse_args()


class Transforms:
    def __init__(self, preimg_resize, postimg_size, crop_division_num):
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

    # def write_file(file_dir, img=None, boxes=None, classes=None):
    #     img_write_dir = './images/' + file_dir + '.png'
    #     txt_write_dir = './labels/' + file_dir + '.txt'
    #     cv2.imwrite(img_write_dir, img)
    #     with open(txt_write_dir, 'w') as f:
    #         for i in range(len(boxes)):
    #             strs = classes[i] + ' ' + ' '.join(str(s) for s in boxes[i]) + '\n'
    #             f.write(strs)

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
                ls = []
                label.append([int(j[0]), *j[1]])
            labels.append(label)
        return images, labels

    def visualize_bbox(self, img, bboxes, classes, thickness=2, write_text=False):
        """Visualizes a bounding box on the image"""
        assert img is not None, 'image is empty'
        BOX_COLOR = {0: (255, 255, 000), 1: (102, 255, 153), 2: (10, 50, 100), 3: (255, 255, 255), 4: (255, 000, 255)}

        img_size = img.shape[0]
        for bbox, cls in zip(bboxes, classes):
            x_center, y_center, x_w, y_h = bboxes
            x_min, x_max, y_min, y_max = int((x_center-(1/2)*x_w) * img_size), int((x_center+(1/2)*x_w) * img_size), int((y_center-(1/2)*y_h) * img_size), int((y_center+(1/2)*y_h) * img_size)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR[int(cls)%5], thickness=thickness)

            if write_text:
                cv2.putText(img, str(cls), (x_min, y_max), 1, color=BOX_COLOR[int(cls)%5], thickness=2)

        return img







class Cell_transform(Transforms):
    def __init__(self, preimg_resize, postimg_size, crop_division_num):
        super().__init__(preimg_resize, postimg_size, crop_division_num)
        self.cwd = os.getcwd()
        self.folder_dir = args.folder_dir
        self.image_root_dir = os.path.join(self.cwd, self.folder_dir, 'images')
        self.label_root_dir = os.path.join(self.cwd, self.folder_dir, 'labels')
        self.image_listdir = os.listdir(self.image_root_dir)
        self.label_listdir = os.listdir(self.label_root_dir)
        self.image_dirs = sorted([os.path.join(self.image_root_dir, img_lst) for img_lst in self.image_listdir])
        self.label_dirs = sorted([os.path.join(self.label_root_dir, img_lst) for img_lst in self.label_listdir])
        self.image_extension = '.png'
        self.label_extension = '.txt'

    def find_mean_std(self, img_lsts):
        meanRGB = [np.mean(x, axis=(0, 1)) for x in img_lsts]
        stdRGB = [np.std(x, axis=(0, 1)) for x in img_lsts]

        meanR = np.mean([m[0] for m in meanRGB])
        meanG = np.mean([m[1] for m in meanRGB])
        meanB = np.mean([m[2] for m in meanRGB])

        stdR = np.mean([s[0] for s in stdRGB])
        stdG = np.mean([s[1] for s in stdRGB])
        stdB = np.mean([s[2] for s in stdRGB])

        return (meanR, meanG, meanB), (stdR, stdG, stdB)

    def find_label_quadrate(self, labels):
        new_label = []
        for label in labels:
            if label[-1] / label[-2] > 0.85 and label[-1] / label[-2] < 1.20:
                new_label.append(label)
        return new_label


    def write_crop_img_label(self):
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(cwd, self.folder_dir, 'crop_images')):
            os.mkdir(os.path.join(cwd, self.folder_dir, 'crop_images'))


        if not os.path.exists(os.path.join(cwd, self.folder_dir, 'crop_labels')):
            os.mkdir(os.path.join(cwd, self.folder_dir, 'crop_labels'))

        for i, (image_dir, label_dir) in enumerate(zip(self.image_dirs, self.label_dirs)):
            image = self.get_img(image_dir)
            label = self.get_label(label_dir)
            crop_images, crop_labels = self.img_label_crop(image, *label)
            new_crop_labels = []
            for k in crop_labels:
                crop_label = self.find_label_quadrate(k)
                new_crop_labels.append(crop_label)
            for j, crop_image in enumerate(crop_images):
                cv2.imwrite(os.path.join(os.path.join(cwd, self.folder_dir, 'crop_images'), os.path.basename(image_dir).split('.')[0]) +'_'+ str(j) + self.image_extension, crop_image)

                with open(os.path.join(os.path.join(cwd, self.folder_dir, 'crop_labels'), os.path.basename(label_dir).split('.')[0]) +'_' + str(j) + self.label_extension, 'a') as f:
                    for line in crop_labels:
                        f.write(' '.join(str(s) for s in line)+'\n')








if __name__ == '__main__':
    img_trans = Cell_transform(args.preimg_resize, args.postimg_resize, args.crop_division_num)
    img_trans.write_crop_img_label()

