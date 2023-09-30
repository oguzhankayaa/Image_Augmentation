from ultralytics import YOLO
import cv2
from PIL import Image
import random
from shapely.geometry import Polygon
import math
import numpy as np
from matplotlib.path import Path
import os
model = YOLO('yolov8l-seg.pt')

def pointGenCrop(ratio,img):
    originalHeight=img.shape[0]
    originalWidth=img.shape[1]
    originalSize=[img.shape[1],img.shape[0]]
    cropWidth=int(ratio*originalWidth)
    cropHeight=int(ratio*originalHeight)
    bottomRight=(originalWidth-cropWidth,originalHeight-cropHeight)
    #print(bottomRight)
    start_x = random.randint(0, bottomRight[0])
    start_y = random.randint(0, bottomRight[1])

    end_x = start_x + cropWidth
    end_y = start_y + cropHeight
    #print("start", start_x, start_y,"crop",cropWidth,cropHeight, "end", end_x, end_y)
    p1 = (start_x / originalWidth, start_y / originalHeight)
    p2 = (end_x / originalWidth, start_y / originalHeight)
    p3 = (end_x / originalWidth, end_y / originalHeight)
    p4 = (start_x / originalWidth, end_y / originalHeight)

    s1 = (start_x / originalWidth, start_y / originalHeight)
    s2 = (start_x / originalWidth, end_y / originalHeight)
    s3 = (end_x / originalWidth, start_y / originalHeight)
    s4 = (end_x / originalWidth, end_y / originalHeight)

    #print("points: ",p1,p2,p3,p4)
    return p1,p2,p3,p4,start_x,start_y,end_x,end_y,cropWidth,cropHeight,originalSize



def writeCrop(id,coords,origin,crop,txt,start_x,start_y):

    norm=[]
    for point in coords:
        normalized_x = ((point[0] * origin[0])-start_x)/crop[0]
        normalized_y = ((point[1] * origin[1])-start_y)/ crop[1]
        norm.append((normalized_x, normalized_y))
    with open(txt, 'a') as file:
        file.write(f"{id}")
        for point in norm:
            file.write(f" {point[0]} {point[1]}")
        file.write("\n")


def crop(img_ratio,txt_path,image,output_path,scratch_ratio=0.8):
    img = cv2.imread(image)

    coords = []
    with open(txt_path, 'r') as file:

        for line in file:
            point = []
            parts = line.strip().split()
            class_id = int(parts[0])
            i = 1
            while i + 1 < len(parts):
                point.append((float(parts[i]), float(parts[i + 1])))
                i += 2
            coords.append(point)
    scratch = Polygon(coords[0])

    i = 0
    while i < 5:
        p1, p2, p3, p4, start_x, start_y, end_x, end_y, width, height, originalSize = pointGenCrop(img_ratio,img)
        cropSize = [width, height]
        points = [p1, p2, p3, p4]
        promt = Polygon(points)
        scratch_ar = scratch.area

        if promt.intersection(scratch):
            inter_ar = promt.intersection(scratch).area
            if ((inter_ar / scratch_ar) > scratch_ratio):
                cropped = img[start_y:end_y, start_x:end_x]
                cv2.imwrite(output_path+ str(i) + ".jpg", cropped)
                for temp in coords:
                    scratch = Polygon(temp)
                    inter = promt.intersection(scratch).exterior.coords
                    writeCrop(class_id, inter, originalSize, cropSize, output_path + str(i) + ".txt", start_x, start_y)
                i += 1



#rotation

def rotate_image(img, angle,output_path):

    # Rotate the image
    rotated_img = img.rotate(angle, expand=False)

    rotated_img.save(output_path+".jpg")

    return rotated_img

def mult(angle, point,origin):

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return  int(qx),int( qy)




def writeRot(coords, id, txt, width, height):
    with open(txt, 'a') as file:
        if len(coords)>1:
            file.write(f"{id}")
            for i, point in enumerate(coords):
                if i < len(coords) - 1:  # Exclude the last coordinate
                    file.write(f" {point[0] / width} {point[1] / height}")
            file.write("\n")




def rotate(angle,txt_path,width,height,output_path):

    rad=math.radians(angle)
    rotatedTopLeft=mult(rad,(0,0),(width/2,height/2))
    rotatedBottomRight=mult(rad,(width,height),(width/2,height/2))
    rotatedFrame = [
        (rotatedTopLeft[0], rotatedTopLeft[1]),
        (rotatedTopLeft[0], rotatedBottomRight[1]),
        (rotatedBottomRight[0], rotatedBottomRight[1]),
        (rotatedBottomRight[0], rotatedTopLeft[1])
    ]
    coords = []
    rotatedCoords=[]
    with open(txt_path, 'r') as file:

        for line in file:
            point = []
            parts = line.strip().split()
            class_id = int(parts[0])
            i = 1
            while i + 1 < len(parts):
                point.append((float(parts[i]), float(parts[i + 1])))
                i += 2
            coords.append(point)
            #print("point:",point)
    for points in coords:
        rotatedCoords=[]
        for point in points:
            rotatedCoords.append((mult(rad,(point[0]*width,point[1]*height),(width/2,height/2))))
        framePoly = Polygon(rotatedFrame)
        temp = Polygon(rotatedCoords).convex_hull
        actual_frame = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        intersection_polygon = actual_frame.intersection(temp)



        finalScratch = intersection_polygon.exterior.coords
        finalScratch_normalized = [(x / width, y / height) for x, y in finalScratch]

        writeRot(finalScratch,class_id,output_path+".txt",width,height)



def executeRotation(image,angle,out,txt):
    img = Image.open(image)
    width, height = img.size
    rotate_image(img,angle,out)
    rotate(-angle,txt,width,height,out)






#color

#coordinate values between 0 and 1
def getNormCoord(txt_path):
    coords = []
    with open(txt_path, 'r') as file:

        for line in file:
            point = []
            parts = line.strip().split()
            class_id = int(parts[0])
            i = 1
            while i + 1 < len(parts):
                point.append((float(parts[i]), float(parts[i + 1])))
                i += 2
            coords.append(point)
    return coords
#pixel coordinates
def getPixelCor(coordinates,height,width):
    got=[]
    for (x,y) in coordinates:
        got.append((int(x*width),int(y*height)))
    return got

def find_most_common_color(image_path, polygon_vertices, color_intervals):
    img = Image.open(image_path)
    width, height = img.size
    pixels = img.load()

    polygon_path = Path(polygon_vertices)

    color_counts = {color: 0 for color in color_intervals}

    for y in range(height):
        for x in range(width):
            if polygon_path.contains_point((x, y)):
                r, g, b = pixels[x, y]
                for color, interval in color_intervals.items():
                    if interval[0] <= r <= interval[1] and \
                            interval[2] <= g <= interval[3] and \
                            interval[4] <= b <= interval[5]:
                        color_counts[color] += 1
    print(color_counts)
    most_common_color = max(color_counts, key=color_counts.get)
    return most_common_color


def whiteChange(img, width, height, polygon_path, wg, pixels, color_intervals, output_path,image_name,mostcom):
    i = 0

    while i < wg:
        m1 = random.random()
        m2 = random.random()
        m3 = random.random()

        new_img = img.copy()  # Create a copy of the original image

        for y in range(height):
            for x in range(width):
                if polygon_path.contains_point((x, y)):
                    r, g, b = pixels[x, y]
                    for color, interval in color_intervals.items():
                        if interval[0] - 100 <= r <= interval[1] + 100 and \
                                interval[2] - 100 <= g <= interval[3] + 100 and \
                                interval[4] - 100 <= b <= interval[5] + 100:
                            if color == mostcom:
                                new_img.putpixel((x, y), (int(r * m1), int(g * m2), int(b * m3)))
        new_img.save(os.path.join(output_path, image_name+str(i + 1)+'.png'))
        i += 1
    img.show()

def convert_to_color_orders(img, polygon_path,output_path,image_name):
    img_array = np.array(img)

    # Create a dictionary to map channel orders to their indices
    channel_orders = {
        'RGB': [0, 1, 2],
        'RBG': [0, 2, 1],
        'BGR': [2, 1, 0],
        'BRG': [2, 0, 1],
        'GBR': [1, 2, 0],
        'GRB': [1, 0, 2]
    }
    i=0
    # Iterate through each channel order and convert the image
    for order_name, order_indices in channel_orders.items():
        converted_img_array = img_array.copy()

        for y in range(img.height):
            for x in range(img.width):
                if polygon_path.contains_point((x, y)):
                    r, g, b = img_array[y, x]

                    # Rearrange the values using order_indices
                    new_color = [0, 0, 0]  # Initialize with zeros
                    new_color[order_indices[0]] = r
                    new_color[order_indices[1]] = g
                    new_color[order_indices[2]] = b

                    converted_img_array[y, x] = new_color
        converted_img = Image.fromarray(converted_img_array)
        converted_img.save(os.path.join(output_path, image_name+str(i + 1)+'.png'))
        i+=1


def change_most_common_color(image_path, polygon_vertices, color_intervals,output_path,image_name,wg=1):
    img = Image.open(image_path)
    width, height = img.size
    pixels = img.load()

    polygon_path = Path(polygon_vertices)

    most_common_color = find_most_common_color(image_path, polygon_vertices, color_intervals)
    print(most_common_color)
    if most_common_color == 'white':
        whiteChange(img,width,height,polygon_path,wg,pixels,color_intervals,output_path,image_name,most_common_color)
    elif most_common_color != 'black':
        convert_to_color_orders(img,polygon_path,output_path,image_name)
        #whiteChange(img,width,height,polygon_path,wg,pixels,color_intervals,output_path,image_name,most_common_color)


def lastcolor(model,image_path,output_path,wg=1):
    color_intervals = {
        'black': (0, 15, 0, 15, 0, 15),
        'green': (0, 50, 50, 150, 0, 50),
        'white': (175, 255, 175, 255, 175, 255),
        'red': (10, 255, 0, 50, 0, 50),
        'blue': (0, 50, 0, 80, 120, 255),
        'orange': (150, 255, 50, 150, 0, 50)
    }
    full_filename = os.path.basename(image_path)
    image_name = os.path.splitext(full_filename)[0]
    image = cv2.imread(image_path)
    height, width = image.shape[0], image.shape[1]
    result = model(source=image, classes=[2], show_labels=False, show=True, line_width=1, save_txt=True,
                   save_conf=True,conf=0.5,iou=0.8)
    save_path = result[-1].save_dir
    txt_path = save_path + r"\labels\image0.txt"
    a = getNormCoord(txt_path)  # D:\PYTHON\demo1\Yolo task3\runs\segment\predict7\labels\image0.txt
    t = getPixelCor(a[0], height, width)
    change_most_common_color(image_path, t, color_intervals,output_path,image_name,wg)





###########EXAMPLE######################

#crop
#crop fonksiyonu ana fonksiyon

cropim=r"s.jpg"
croptxt=r"s.txt"
full_filename = os.path.basename(cropim)
img_name = os.path.splitext(full_filename)[0]
cropout = os.path.join(r"D:\PYTHON\demo1\Yolo task3\cropped", img_name + "_cropped")
img_ratio=0.5
crop_rat=0.8 #opsiyonel scratchin en az % kaçını alacağını gösterir

#crop(img_ratio,croptxt,cropim,cropout,crop_rat)


#rotation
rotim=r"D:\PYTHON\demo1\Yolo task3\torot\1164853167_right.jpeg"
rotxt=r"D:\PYTHON\demo1\Yolo task3\torot\1164853167_right.txt"
full_filename = os.path.basename(rotim)
image_name = os.path.splitext(full_filename)[0]
rotout = os.path.join(r"D:\PYTHON\demo1\Yolo task3\rotout", image_name + "_rotated")
ang=-50
#executeRotation(rotim,ang,rotout,rotxt)



#color
image_path = r"1165427507_left.jpeg"
out="D:\PYTHON\demo1\Yolo task3\colored"
#lastcolor(model,image_path,out,20)
