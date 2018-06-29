import cv2
import numpy as np
import time
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randint

import multiprocessing
print(multiprocessing.cpu_count())
DATASET = '../../Apollo/test/'
#DATASET = '/home/shan/Dataset/Apollo/test/'
df  = pd.read_csv('../csv/1560.csv')#result_0607_3384.csv  result_0607_full.csv
df2 = pd.read_csv('../csv/sample_submission.csv') 
Image_Ids   = df2['ImageId'].tolist()
Image_eval  = df['ImageId'].tolist()

cnt = 0
label_name   = ['background', 'person','bicycle','car','bus','truck','tricycle','rider', 'motorcycle']
label_color  = [(20,20,20),(180,20,20),(20,180,20),(20,20,180),(180,180,20),(20,180,180),(128,128,180),(200,128,128),(180,20,180)]

label_list   = [0, 36, 35, 33, 39, 38, 40, 37, 34]

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

for image_id in Image_Ids:
    #try:
        idxs = indices(Image_eval, image_id)
        #print(idxs)
        img  = cv2.imread(DATASET + str(image_id) + '.jpg')
        #print((np.array(img)).shape)
        cnt = cnt +1

        if (len(idxs)>0) :
            for i in range (0, len(idxs)):
                
                pixel_mask = df['EncodedPixels'][idxs[i]]
                label_id   = df['LabelId'][idxs[i]]
                color_id = label_list.index(label_id)
                seg = pixel_mask.split('|')
                #print(seg)
                if (df['Confidence'][idxs[i]]>0.0):
                    x_min = 10000
                    y_min = 10000
                    for k in range (0, len(seg)-1):
                        coorindate = int(seg[k].split(' ')[0])
                        y =  int ( int(coorindate) / 3384 )
                        x =  int (coorindate) - 3384*y   
                        if (x_min>x):
                            x_min = x
                        if (y_min>y):
                            y_min = y
                        length    = int (seg[k].split(' ')[1]) 
                        img[y,x:x+length, 0] =  label_color[color_id][0]# + ((i+4)%4)*5
                        img[y,x:x+length, 1] =  label_color[color_id][1]# - ((i+4)%4)*5
                        img[y,x:x+length, 2] =  label_color[color_id][2]# + ((i+4)%4)*5
                #cv2.putText(img,str(round(df['Confidence'][idxs[i]],2)),(x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1, 1)
            xx = 0
            for idx in range(1,9):      
                cv2.rectangle(img,(xx,0),(128+xx,128),label_color[idx],-1)
                cv2.putText(img,label_name[idx],(xx,200), cv2.FONT_HERSHEY_SIMPLEX, 2,label_color[idx],2, 2)
                xx = xx + 256
            cv2.imwrite('./s1/' + str(image_id) + '.jpg', img)
            print(image_id, cnt)

    #except ValueError:
    #    print (image_id, cnt)
    #    cnt = cnt + 1
        print("==========")
