import cv2
import numpy as np
import scipy as sp
import math
from math import *
from PIL import Image, ImageFilter
from pptx import Presentation
from pptx.util import Inches
from pptx.util import Mm
from pptx.util import Cm
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
import matplotlib.pyplot as plt
import matplotlib as mpl

#def sobel_filters(img):
#    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
#    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
#    
#    Ix = ndimage.filters.convolve(img, Kx)
#    Iy = ndimage.filters.convolve(img, Ky)
#    
#    G = np.hypot(Ix, Iy)
#    G = G / G.max() * 255
#    theta = np.arctan2(Iy, Ix)
#    
#    return (theta*180/3.14)

#def get_gradient(image, kernel_size):
#    
#    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0 ,ksize=kernel_size)
#    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1 ,ksize=kernel_size)
#    
#    grad = grad_x + 1j*grad_y
#   
#    return grad

# Load Yolo
net = cv2.dnn.readNet("C:/Users/Administrator/darknet-master/build/darknet/x64/backup/yolo-obj_last.weights", "C:/Users/Administrator/darknet-master/build/darknet/x64/data/yolo-obj.cfg")
classes = []                ##weight file
mpl.rcParams["figure.dpi"] = 300
with open("C:/Users/Administrator/darknet-master/build/darknet/x64/data/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
roi=0
# Loading image
img = cv2.imread("C:/Users/Administrator/textsample/a77.jpg")
img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#Creating PowerPoint
presentation = Presentation()
title_slide_layout = presentation.slide_layouts[6]
slide = presentation.slides.add_slide(title_slide_layout)
shapes = slide.shapes

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
list = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[1]
        if label == "text":  
            #여기에 roi를 tesseract에 보내면 됨
            dst = img.copy() #원본 img copy
            dst = img[y - 7 : y + h + 7 , x-10 : x + w + 10]
            roi = dst
            #image grayscale
            roi = cv2.cvtColor(np.asarray(roi), cv2.COLOR_BGR2GRAY)
            roi_pil = Image.fromarray(roi)
            roi_np = np.asarray(roi_pil)
            s=str(i)
            
                #Otsu threshold & Gaussian blur
            #ret1, th1 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #blur = cv2.GaussianBlur(roi, (1,1), 0)
            
                #AdaptiveThreshold & Gaussian blur
            blur= cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 13)
            
                #erode & dilate
            #kernel = np.ones((3,3), np.uint8)
            #blur= cv2.erode(blur, kernel, 1)
            #kernel = np.ones((3,3), np.uint8)
            #blur = cv2.dilate(blur, kernel, 1)
            
            #result = cv2.resize(result, None, fx=0.2, fy=0.2)
            cv2.imwrite('C:/Users/Administrator/textimage/7' + s + '.jpg',blur) #label == text일때 이미지저장
            
            #dpi 조절
            source_image = 'C:/Users/Administrator/textimage/7' + s + '.jpg'
            image = Image.open(source_image)
            image = image.filter(ImageFilter.SHARPEN)
            image.save('C:/Users/Administrator/Result/7' + s + '.jpg', dpi = (400,400))
                     
        if label == "circle":
            
            shape = shapes.add_shape(MSO_SHAPE.OVAL, Cm(x)/50, Cm(y)/50 , Cm(w)/50, Cm(h)/50)
            shape.fill.background()
            shape.shadow.inherit = False
            line = shape.line
            line.color.rgb = RGBColor(0,0,0)
            #line.color.brightness = 0.5
            #line.width = Mm(1)
        
            
        if label == "rectangle":
            shape = shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(x)/50, Cm(y)/50 , Cm(w)/50, Cm(h)/50)
            shape.fill.background()
            shape.shadow.inherit = False
            line = shape.line
            line.color.rgb = RGBColor(0,0,0)
            #line.color.brightness = 0.5
           # line.width = Mm(1)
            
            
        if label == "triangle":
            shape = shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Cm(x)/50, Cm(y)/50 , Cm(w)/50, Cm(h)/50)
            shape.fill.background()
            shape.shadow.inherit = False
            line = shape.line
            line.color.rgb = RGBColor(0,0,0)
            #line.color.brightness = 0.5
           # line.width = Mm(1)
            
            
        if label == "pentagon":
            shape = shapes.add_shape(MSO_SHAPE.REGULAR_PENTAGON, Cm(x)/85, Cm(y)/140 , Cm(w)/180, Cm(h)/180)
            shape.fill.background()
            shape.shadow.inherit = False
            line = shape.line
            line.color.rgb = RGBColor(0,0,0)
            #line.color.brightness = 0.5
           # line.width = Mm(1)

        
        #수정해야되는부분
        if label == "arrow1":
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

            img = img.copy() #원본 img copy
            gray = img[y - 9 : y + h + 9 , x-13 : x + w + 13]
            mask =  np.zeros_like(gray)
            mask1 = np.zeros_like(gray)
            ret1, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
            ret2, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY_INV)
            
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            ret, gray = cv2.threshold(gray, 250, 255, cv2.THRESH_OTSU)
            
            image_binary, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)           
            
            #drawContour on mask
            cnt= sorted(contours, key = cv2.contourArea, reverse=True)[1]
            cv2.drawContours(mask, cnt, -1, (0,0,0), 3)
            
            #Ramer-Douglas-Peucker algorithm
            epsilon = 0.02 * cv2.arcLength(cnt,True)
            approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(mask1, approx_corners, -1, (0,0,0), 6)
            
            approx_corners = sorted(np.concatenate(approx_corners).tolist())
            approx_corners = [approx_corners[i] for i in [0, 1, 2, 3, 4, 5, 6]]
            
            #무게중심
            value_x = 0
            value_y = 0
            for i in approx_corners:
                a,b = np.ravel(i)
                value_x = value_x + a
                value_y = value_y + b
            
            centerx = value_x/7
            centery = value_y/7
            print(approx_corners)
            print("centerx : ", centerx)
            print("centery : ", centery)
            
            #무게중심에서 가장 가까운 두 점 계산
            min_distancex1 = 0
            min_distancey1 = 0
            min_distance1 = 1000000
            for i in approx_corners:
                a,b = np.ravel(i)
                distance = math.sqrt(math.pow((centerx-a),2) + math.pow((centery-b),2))
                if distance < min_distance1:
                    min_distance1 = distance
                    min_distancex1 = a
                    min_distancey1 = b
            
            distance1 = np.array([min_distancex1, min_distancey1])
            
            print("min_distancex1 : ", min_distancex1)
            print("min_distancey1 : ", min_distancey1)
               
            min_distancex2 = 0
            min_distancey2 = 0
            min_distance2 = 100000
            for i in approx_corners:
                a,b = np.ravel(i)
                distance = math.sqrt(math.pow((centerx-a),2) + math.pow((centery-b),2))
                if distance < min_distance2:
                    if a==min_distancex1 and b==min_distancey1:
                        continue
                    min_distance2 = distance
                    min_distancex2 = a
                    min_distancey2 = b
                    
            distance2 = np.array([min_distancex2, min_distancey2])
            
            print("min_distancex2 : ", min_distancex2)
            print("min_distancey2 : ", min_distancey2)
            
            center = np.array([centerx,centery])
            
            z1 = [i-j for i,j in zip(distance1, center)]
            z2 = [i-j for i,j in zip(distance2, center)]
            
            print("z1 : ", z1)
            print("z2 : ", z2)
            
            #벡터 연산
            vector = [i+j for i,j in zip(z1,z2)]
            a, b = np.ravel(vector)
            
            print("vector : ", vector)
            
            reference = np.array([1,0])
            vector = np.array([a,-b])
        
            
            vector_size = math.sqrt(math.pow((a),2) + math.pow((b),2))
            cosinseta = np.dot(vector, reference) / vector_size
            print("cosinseta : ", cosinseta)
            

            degree = np.arccos(cosinseta)
            degree = np.rad2deg(degree)
            print("degree : ", degree)
            
            shape = shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Cm(x)/50, Cm(y)/50 , Cm(w)/50, Cm(h)/50)
            #rotate clockwise
            
            shape.rotation = - degree
            
            shape.fill.background()
            shape.shadow.inherit = False
            line = shape.line
            line.color.rgb = RGBColor(0,0,0)
            
                         
             #Shi-Tomasi corner detection algorithm
            #corners = cv2.goodFeaturesToTrack(mask, 7, 0.01, 4)
            #corners = np.int0(corners)
            #for i in corners:
            #    a,b = i.ravel()
            #    cv2.circle(mask1, (a,b), 2, 255, -1)
              
             #Approxmate contour algorithm
            #for cnt in contours:
            #    epsilon = 0.001* cv2.arcLength(cnt, True)
            #    approx = cv2.approxPolyDP(cnt, epsilon, True)
            #    cv2.drawContours(mask, [approx], -1, (0,0,0), 3)

   
            #ret, img_binary = cv2.threshold(gray, 127, 255, 0)
            #img_binary, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            
            
            s=str(i)
            cv2.imwrite('C:/Users/Administrator/Result/1' + s + '.jpg',mask)
            cv2.imwrite('C:/Users/Administrator/Result/2' + s + '.jpg',mask1)
            
        if label == "arrow2":
            
            img = img.copy() #원본 img copy
            gray = img[y - 9 : y + h + 9 , x-13 : x + w + 13]
            mask =  np.zeros_like(gray)
            mask1 = np.zeros_like(gray)
            ret1, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
            ret2, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY_INV)
            
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            ret, gray = cv2.threshold(gray, 250, 255, cv2.THRESH_OTSU)
            
            image_binary, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)           
            
            #drawContour on mask
            cnt= sorted(contours, key = cv2.contourArea, reverse=True)[1]
            cv2.drawContours(mask, cnt, -1, (0,0,0), 3)
            
            
            #Ramer-Douglas-Peucker algorithm
            epsilon = 0.062 * cv2.arcLength(cnt,True)
            approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(mask1, approx_corners, -1, (0,0,0), 6)
            
            approx_corners = sorted(np.concatenate(approx_corners).tolist())
            approx_corners = [approx_corners[i] for i in [0, 1, 2, 3, 4]]
            
            #무게중심
            value_x = 0
            value_y = 0
            for i in approx_corners:
                a,b = np.ravel(i)
                value_x = value_x + a
                value_y = value_y + b
            
            centerx = value_x/5
            centery = value_y/5
            
            
            print(approx_corners)
            print("centerx : ", centerx)
            print("centery : ", centery)
            
            #무게중심에서 가장 가까운 두 점 계산
            min_distancex1 = 0
            min_distancey1 = 0
            min_distance1 = 1000000
            for i in approx_corners:
                a,b = np.ravel(i)
                distance = math.sqrt(math.pow((centerx-a),2) + math.pow((centery-b),2))
                if distance < min_distance1:
                    min_distance1 = distance
                    min_distancex1 = a
                    min_distancey1 = b
            
            real_distance1 = math.sqrt(math.pow((centerx-min_distancex1),2) + math.pow((centery-min_distancey1),2))
            distance1 = np.array([min_distancex1, min_distancey1])
            
            print("min_distancex1 : ", min_distancex1)
            print("min_distancey1 : ", min_distancey1)
               
            min_distancex2 = 0
            min_distancey2 = 0
            min_distance2 = 100000
            min_difference = 100000
            difference = 0
            for i in approx_corners:
                a,b = np.ravel(i)
                distance = math.sqrt(math.pow((centerx-a),2) + math.pow((centery-b),2))
                difference = abs(distance - real_distance1)
                if difference < min_difference:
                    if a==min_distancex1 and b==min_distancey1:
                        continue
                    min_distance2 = distance
                    min_difference = difference
                    min_distancex2 = a
                    min_distancey2 = b
                    print("difference : ", difference)
                    
            distance2 = np.array([min_distancex2, min_distancey2])
            
            print("min_distancex2 : ", min_distancex2)
            print("min_distancey2 : ", min_distancey2)
            
            center = np.array([centerx,centery])
            
            z1 = [i-j for i,j in zip(distance1, center)]
            z2 = [i-j for i,j in zip(distance2, center)]
            
            print("z1 : ", z1)
            print("z2 : ", z2)
            
            #벡터 연산
            vector = [i+j for i,j in zip(z1,z2)]
            a, b = np.ravel(vector)
            
            print("vector : ", vector)
            
            reference = np.array([1,0])
            vector = np.array([a,-b])
        
            
            vector_size = math.sqrt(math.pow((a),2) + math.pow((b),2))
            cosinseta = np.dot(vector, reference) / vector_size
            print("cosinseta : ", cosinseta)
            

            degree = np.arccos(cosinseta)
            degree = np.rad2deg(degree)
            print("degree : ", degree)
            
            
            shape = shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Cm(x)/50, Cm(y)/50 , Cm(w)/50, Cm(h)/50)
            
            #rotate clockwise
            shape.rotation = -degree
            
            shape.fill.background()
            shape.shadow.inherit = False
            line = shape.line
            line.color.rgb = RGBColor(0,0,0)
            
            
            s=str(i)
            cv2.imwrite('C:/Users/Administrator/Result/3' + s + '.jpg',mask)
            cv2.imwrite('C:/Users/Administrator/Result/4' + s + '.jpg',mask1)
            

            #convexHull algorithm
        #    gray_image = cv2.GaussianBlur(gray_image, (11,11), 3)
        #    
        #    
        #    #Convex Hull
        #    ret, img_binary = cv2.threshold(gray_image, 127, 255, 0)
        #    img_binary, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #   
        #    
        #    for cnt in contours:
        #        hull = cv2.convexHull(cnt)
        #        cv2.drawContours(gray,[hull], 0, (255,0,255), 6)
        #        

        #    for cnt in contours:
        #        hull = cv2.convexHull(cnt, returnPoints = False)
        #        defects = cv2.convexityDefects(cnt, hull)

        #        for i in range(defects.shape[0]):
        #            s,e,f,d = defects[i,0]
        #            start = tuple(cnt[s][0])
        #            end = tuple(cnt[e][0])
        #           far = tuple(cnt[f][0])
        #            
        #            print(d)
        #            
        #            cv2.circle(gray, far, 5, (0,255,0),-1)
        #   
        #    s=str(i)
        #    cv2.imwrite('C:/Users/Administrator/Result/1' + s + '.jpg',gray)
    
presentation.save('C:/Users/Administrator/test.pptx')

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
outs = net.forward(output_layers)