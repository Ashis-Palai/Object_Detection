from paddleocr import PaddleOCR,draw_ocr
import pandas as pd
import torch
import torchvision
from glob2 import glob
import numpy as np
import cv2



from paddleocr import PaddleOCR,draw_ocr
import pandas as pd
import torch
import torchvision
from glob import glob
import numpy as np
import cv2


# DEVICE AND MODEL INITIALIZATION

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
SSD_MODEL = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=10) # DEFINING SSD MODEL WITH TOTAL CLASS INCLUDING BACKGROUNG CLASS
checkPoint = torch.load("logo_detection_ckpt_SSD_MODEL.pth",map_location = device)
SSD_MODEL.load_state_dict(checkPoint['model_state_dict'])
SSD_MODEL.eval()
SSD_MODEL.to(device)

#VARIABLE INITIALIZATION

symbol_list = []
device_Name_list = []
REF_list = []
LOT_list = []
Qty_list = []

ocr = PaddleOCR(use_angle_cls=True, lang='en') # NEED TO RUN ONLY ONCE TO DOWNLOAD AND LOAD MODEL INTO MEMORY


#READING IMAGES FROM THE FOLDER

for image in glob("images/*"):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img/255.0
    img_tensor_float = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0).to(device)
    out = SSD_MODEL(img_tensor_float)
    vertical_list = []
    horizontal_list = []

    for i , j , k in zip((out[0]['boxes']).detach().numpy(), (out[0]['scores']).detach().numpy(), (out[0]['labels']).detach().numpy()):
        if j > 0.5 :
          if (i[0] -i[1]) >800 :
            vertical_list.append((i[0],k))
          else:
            horizontal_list.append((i[0],i[1],k))

    vertical_list.sort()
    horizontal_list.sort()

#LOGIC TO READ THE IMAGES FROM TOP TO BOTTOM

    for i in range(len(horizontal_list)):
        if i < len(horizontal_list)-1 and horizontal_list[i+1][1] -horizontal_list[i][1] >60 :
            odd = (horizontal_list[i+1])
            horizontal_list.pop(i+1)
            horizontal_list.append(odd)
    vertical_list = [i[1] for i in vertical_list]
    horizontal_list = [i[2] for i in horizontal_list]
    temp = [vertical_list.append(i) for i in horizontal_list]
    all_symb = ''
    for i in vertical_list:
        all_symb = all_symb+ ''+ str(i)
    symbol_list.append(all_symb)

# READING TEXT FROM IMAGES USING PADDLE-OCR

    result = ocr.ocr(image, cls=True)
    for i in (result[0]):
        if i[1][0].startswith("Device Name:"):
            before_keyword1, keyword1, Device_name = i[1][0].partition('Device Name:')
            device_Name_list.append(Device_name)
        elif i[1][0].startswith("REF"):
            before_keyword2, keyword2, REF = i[1][0].partition('REF')
            REF_list.append(REF)
        elif i[1][0].startswith("LOT:"):
            before_keyword3, keyword3, LOT = i[1][0].partition('LOT:') 
            LOT_list.append(LOT)
        elif i[1][0].startswith("Qty:"):
            before_keyword4, keyword4, Qty = i[1][0].partition('Qty:')
            Qty_list.append(Qty)

#CREATING DATAFRAME TO SAVE IN EXCEL FILE

data = pd.DataFrame(columns = ['Device Name','REF','LOT','Qty','Symbols'])
data['Device Name'] = device_Name_list
data['REF'] = REF_list
data['LOT'] = LOT_list
data['Qty'] = Qty_list
data['Symbols'] = symbol_list
data.to_excel("output.xlsx",index=False)
print('Outputfile in Excel format generated and saved')
