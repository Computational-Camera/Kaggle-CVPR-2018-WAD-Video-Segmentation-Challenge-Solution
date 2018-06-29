import cv2
import numpy as np
import time
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
 

CSV_FILE = sys.argv[1] #../40.csv'

df1   = pd.read_csv(CSV_FILE)


ImageIds      = df1['ImageId'].tolist()
filenames     = df1['ImageId'].tolist()
labels        = df1['LabelId'].tolist()  
pix_cnt       = df1['PixelCount'].tolist() 
conf          = df1['Confidence'].tolist() 
encode_pixel  = df1['EncodedPixels' ].tolist()
 
r_list = []
ids = []
for idx in  range(0,len(conf)):   
    #if (df1['LabelId'][idx]!=38):
    #    r_list.append(idx)
    if (df1['Confidence'][idx]<=0.25):
        r_list.append(idx) 
    elif (df1['LabelId'][idx]==37): #remove rider
        r_list.append(idx)

    #else:
        #ids.append(filenames[0:-4])
        print(filenames[idx][:-4])
 
for idx in reversed(r_list):
    filenames.pop(idx)
    labels.pop(idx)
    pix_cnt.pop(idx)
    conf.pop(idx)
    encode_pixel.pop(idx)
for idx in range(0, len(filenames)) :
    filenames[idx] = filenames[idx][:-4]
  
df  = pd.DataFrame({ 'ImageId':filenames ,  'LabelId' : labels,  'PixelCount' : pix_cnt, 'Confidence':conf, 'EncodedPixels': encode_pixel}) 
df.to_csv(sys.argv[2], index=False, columns=['ImageId', 'LabelId', 'PixelCount', 'Confidence', 'EncodedPixels'])    
