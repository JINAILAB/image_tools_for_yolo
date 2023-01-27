import albumentations as A
import cv2
import glob

transform1 = A.Compose([
    A.Crop(x_min=0, y_min=0, x_max=1024, y_max=1024)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

transform2 = A.Compose([
    A.Crop(x_min=1024, y_min=0, x_max=2048, y_max=1024)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

transform3 = A.Compose([
    A.Crop(x_min=0, y_min=1024, x_max=1024, y_max=2048)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

transform4 = A.Compose([
    A.Crop(x_min=1024, y_min=1024, x_max=2048, y_max=2048)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

bacteria = { '0' : 'aureus', '1' : 'epimidis'}
img_dirs = glob.glob('./2048_images/*.png')
img_dirs.sort()
txt_dirs = glob.glob('./2048_labels/*.txt')
txt_dirs.sort()



def getfile(img_file_dir, txt_file_dir):
    img = cv2.imread(img_file_dir, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = []
    classes = []
    with open(txt_file_dir, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            bboxes.append(list(map(float, line.split(' ')))[1:])
            classes.append(line.split(' ')[0])
    print(img.shape)
    return img, bboxes, classes

def write_file(file_dir, img, boxes, classes):
    img_write_dir = './images/' + file_dir + '.png'
    txt_write_dir = './labels/' + file_dir + '.txt'
    cv2.imwrite(img_write_dir, img)
    with open(txt_write_dir, 'w') as f:
        for i in range(len(boxes)):
            strs = classes[i] + ' ' + ' '.join(str(s) for s in boxes[i]) + '\n'
            f.write(strs)
    
for i in range(len(img_dirs)):
    img, bboxes, classes = getfile(img_dirs[i], txt_dirs[i])
    transformed1 = transform1(image=img, bboxes=bboxes, class_labels=classes)
    transformed2 = transform2(image=img, bboxes=bboxes, class_labels=classes)
    transformed3 = transform3(image=img, bboxes=bboxes, class_labels=classes)
    transformed4 = transform4(image=img, bboxes=bboxes, class_labels=classes)
    
    dir1 = img_dirs[i].split('.')[-2].split('/')[-1] +'_1'
    dir2 = img_dirs[i].split('.')[-2].split('/')[-1] +'_2'
    dir3 = img_dirs[i].split('.')[-2].split('/')[-1] +'_3'
    dir4 = img_dirs[i].split('.')[-2].split('/')[-1] +'_4'
    
    img_tr1 = transformed1['image']
    bboxes_tr1 = transformed1['bboxes']
    classes_tr1 = transformed1['class_labels']
    
    img_tr2 = transformed2['image']
    bboxes_tr2 = transformed2['bboxes']
    classes_tr2 = transformed2['class_labels']
    
    img_tr3 = transformed3['image']
    bboxes_tr3 = transformed3['bboxes']
    classes_tr3 = transformed3['class_labels']
    
    img_tr4 = transformed4['image']
    bboxes_tr4 = transformed4['bboxes']
    classes_tr4 = transformed4['class_labels']
    
    write_file(dir1, img_tr1, bboxes_tr1, classes_tr1)
    write_file(dir2, img_tr2, bboxes_tr2, classes_tr2)
    write_file(dir3, img_tr3, bboxes_tr3, classes_tr3)
    write_file(dir4, img_tr4, bboxes_tr4, classes_tr4)
    
    
    
    
    
    
    
