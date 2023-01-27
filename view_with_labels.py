import cv2


BOX_COLOR1 = (255, 255, 255) # white
BOX_COLOR2 = (30, 40, 255)


def visualize_bbox(img, bbox, class_num, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    
    if class_num == 0: color = BOX_COLOR1
    elif class_num == 1: color = BOX_COLOR2
    
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)

    return img
