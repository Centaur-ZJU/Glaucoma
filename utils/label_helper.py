import os
import cv2
import json
import tkinter.filedialog as tk


# m_x,m_y,width,height = 0,0,0,0

def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    global m_x,m_y,width,height
    cliped = False

    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 3)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 3)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 3)
        cv2.imshow('image', img2)
        m_x = 1.0*(point1[0] + point2[0])/(2*1024)
        m_y = 1.0*(point1[1] + point2[1])/(2*1024)
        width = 1.0*abs(point1[0] - point2[0])/1024
        height = 1.0*abs(point1[1] - point2[1])/1024


    elif event == cv2.EVENT_MOUSEMOVE and m_x == 0:
        cv2.line(img2,(x,0),(x,1024),(255,255,255),2)
        cv2.line(img2, (0,y), (1024,y), (255, 255, 255), 2)
        cv2.imshow('image', img2)


# dirPath = tk.askdirectory()
# # filePath_ = tk.askopenfilename(title=u'choose file', initialdir=(os.path.expanduser("默认打开路径")))
# files = os.listdir(dirPath)
img_prefix = "D:/Project_Glaucoma/dataset/LAG/"
type =  "non_glaucoma/image"
root = img_prefix + type
files = os.listdir(root)
label = "normal"

output_file = "D:/Project_Glaucoma/dataset/JSONS/LAG_normal0.json"
dataset = []

if os.path.exists(output_file):
    with open(output_file,'r') as f:
        dataset = json.load(f)
exist_num = len(dataset)


with open("D:/Project_Glaucoma/dataset/JSONS/LAG_normal.json",'r') as f:
    labeled_normal = json.load(f)
labeled_names = [normal["img"].split("/")[-1]for normal in labeled_normal]



for index,file in enumerate(files):
    if index<exist_num:
        continue
    #filter img those has been labeled!
    if file in labeled_names:
        continue

    m_x, m_y, width, height = 0, 0, 0, 0
    img = cv2.imread(os.path.join(root,file))
    W,H,_ = img.shape

    img = cv2.resize(img, (1024, 1024))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.putText(img,"id:"+str(file),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1.2, (255,255,255),2)
    cv2.putText(img, label, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    data = {
        "img": type+"/"+file,
        "label": 1 if label is "glaucoma" else 0,
        "bbox": (m_x*W,m_y*H,width*W,height*H),
        "size": (W,H)
    }
    dataset.append(data)
    print(m_x,m_y,width,height,W,H)

    if m_y==0 and m_x==0:
        break


with open(output_file,'w') as f:
    json.dump(dataset,f)






