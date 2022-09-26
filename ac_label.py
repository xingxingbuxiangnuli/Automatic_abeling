import threading
import time
import cv2
import numpy as np
import os

# 将拍摄到的图片保存在data1文件夹中
def func(i,nit):
  print(time.time(),"Hello Timer!",str(i))
  while os.path.exists("data/images"+str(i)+".jpg"):
     i+=1
  cv2.imwrite('data/images/' + str(i) + ".jpg", nit)
  return nit

#自动识别边框 并返回矩阵坐标
def get_coor(nit):
    gray = cv2.cvtColor(nit, cv2.COLOR_BGR2GRAY) #灰度化

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    cnts, _, = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the image cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("Image",img)
    # cv2.imwrite("contoursImage2.jpg", img)
    # cv2.waitKey(0)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    return x1,y1,x2,y2

# 生成xml文件
def save_xml(box_name,src_xml_dir, img_name, h, w, x1, y1, x2, y2):
    xml_file = open((src_xml_dir + '/' + img_name + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>images</folder>\n')

    xml_file.write('    <filename>' + str(img_name) + '.jpg' + '</filename>\n')
    xml_file.write('    <source>\n')
    xml_file.write('    <database>Unknow</database>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(w) + '</width>\n')
    xml_file.write('        <height>' + str(h) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    # box_name = 'Blueberry' #打标签类别名称
    xml_file.write('    <object>\n')
    xml_file.write('        <name>' + box_name + '</name>\n')
    xml_file.write('        <pose>Unspecified</pose>\n')
    xml_file.write('        <truncated>0</truncated>\n')
    xml_file.write('        <difficult>0</difficult>\n')
    xml_file.write('        <bndbox>\n')
    xml_file.write('            <xmin>' + str(x1) + '</xmin>\n')
    xml_file.write('            <ymin>' + str(y1) + '</ymin>\n')
    xml_file.write('            <xmax>' + str(x2) + '</xmax>\n')
    xml_file.write('            <ymax>' + str(y2) + '</ymax>\n')
    xml_file.write('        </bndbox>\n')
    xml_file.write('    </object>\n')
    xml_file.write('</annotation>')

# i = 0
# save = True
# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))  # ?1为啥要重设
# t1=time.time()
# while True:
#     # cap.isOpened(0)
#     ret, frame = cap.read()
#     nit = frame
#     # time.sleep(0.5)
#     # if(save):
#     #     time.sleep(1)
#     #     save = False
#     # t2 = time.time()
#     # if (t2-t1)>=0.5:
#     #   s = threading.Timer(1,func,(i,nit,))
#     #   s.start()
#     #   t1=t2
#     i=i+1
#     func(i,nit)
#
#     x1, y1, x2, y2 = get_coor(nit)
#     draw_0 = cv2.rectangle(frame, (x2,y2), (x1,y1), (255, 0, 0), 2)
#     cv2.imshow('capture', frame)
#     if cv2.waitKey(1)&0xFF==ord('q'):#按键盘q就停止拍照
#         break
# cap.release()
# cv2.destroyAllWindows()
box_name = 'Watermelon' #打标签类别名称
file_dir = r'E:/xing/data/fruit-recognition_datasets/train/train/Watermelon'
save_xml_dir = r'data/xlm/Watermelon'
for name in os.listdir(file_dir):
    print(name)
    img_path = os.path.join(file_dir, name)
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (570,760), interpolation=cv2.INTER_LINEAR)
    # img = cv2.imdecode(np.fromfile(img_path), dtype=np.uint8), -1)
    h, w = img.shape[:-1]
    x1, y1, x2, y2 = get_coor(img)
    img_name = name.split('.')[0]
    save_xml(box_name,save_xml_dir, img_name, h, w, x1, y1, x2, y2)
