from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.


def home(request):
    return render(request, 'Main/home.html')


def canvas(request):
    return render(request, 'Main/canvas.html')


def about(request):
    return render(request, 'Main/about.html')


def output(request):
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    import keras
    import keras.backend.tensorflow_backend as tfback
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K
    keras.backend.image_data_format()
    from keras.models import model_from_json

    string_list_for_OP = []
    # used for errors
    def _get_available_gpus():

        if tfback._LOCAL_DEVICES is None:
            devices = tf.config.list_logical_devices()
            tfback._LOCAL_DEVICES = [x.name for x in devices]
        return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
    tfback._get_available_gpus = _get_available_gpus
    tfback._get_available_gpus()
    tf.config.list_logical_devices()

    # open the model
    json_file = open(
        'C:\\Users\\Abhinav\\Desktop\\MLApp\\Main\\model_final.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(
        "C:\\Users\\Abhinav\\Desktop\\MLApp\\Main\\model_final.h5")

    mdir = 'C:\\Users\\Abhinav\\Downloads\\'
    fil_key = ['.png', 'test']
    newimglist = []
    imglist = os.listdir(str(mdir))
    for img in imglist:
        if fil_key[0] in img and fil_key[1] in img:
            newimglist.insert(0, img)
    print(newimglist)
    num = len(newimglist)
    if(num > 1):
        curimg = newimglist[1]
    else:
        curimg = newimglist[0]

    print(curimg)

    img = cv2.imread(str(mdir)+str(curimg), cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # img=rez(img)
        img = ~img
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ctrs, ret = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w = int(28)
        h = int(28)
        train_data = []
        rects = []
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            rect = [x, y, w, h]
            rects.append(rect)
        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if r[0] < (rec[0]+rec[2]+10) and rec[0] < (r[0]+r[2]+10) and r[1] < (rec[1]+rec[3]+10) and rec[1] < (r[1]+r[3]+10):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)
        dump_rect = []
        for i in range(0, len(cnt)):
            for j in range(0, len(cnt)):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2]*rects[i][3]
                    area2 = rects[j][2]*rects[j][3]
                    if(area1 == min(area1, area2)):
                        dump_rect.append(rects[i])
        final_rect = [i for i in rects if i not in dump_rect]
        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            im_crop = thresh[y:y+h+10, x:x+w+10]

            im_resize = cv2.resize(im_crop, (28, 28))

            im_resize = np.reshape(im_resize, (1, 28, 28))
            train_data.append(im_resize)

    # evaluate
    s = ''
    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i])
        train_data[i] = train_data[i].reshape(1, 1, 28, 28)
        print(train_data[i])
        result = loaded_model.predict_classes(train_data[i])

        if(result[0] == 0):
            s = s+'0'
        if(result[0] == 1):
            s = s+'1'
        if(result[0] == 2):
            s = s+'2'
        if(result[0] == 3):
            s = s+'3'
        if(result[0] == 4):
            s = s+'4'
        if(result[0] == 5):
            s = s+'5'
        if(result[0] == 6):
            s = s+'6'
        if(result[0] == 7):
            s = s+'7'
        if(result[0] == 8):
            s = s+'8'
        if(result[0] == 9):
            s = s+'9'
        if(result[0] == 10):
            s = s+'-'
        if(result[0] == 11):
            s = s+'+'
        if(result[0] == 12):
            s = s+'*'
        if(result[0] == 13):
            s = s+'/'
        if(result[0] == 14):
            s = s+'='
        if(result[0] == 15):
            s = s+'('
        if(result[0] == 16):
            s = s+')'
        if(result[0] == 17):
            s = s+'α'
        if(result[0] == 18):
            s = s+'β'

        if(result[0] == 19):
            s = s+'cos'
        if(result[0] == 20):
            s = s+'sin'
        if(result[0] == 21):
            s = s+'tan'
        if(result[0] == 22):
            s = s+'∫'
        if(result[0] == 23):
            s = s+'∑'
        if(result[0] == 24):
            s = s+'log'
        if(result[0] == 25):
            s = s+'a'
        if(result[0] == 26):
            s = s+'b'
        if(result[0] == 27):
            s = s+'c'
        if(result[0] == 28):
            s = s+'d'

        if(result[0] == 29):
            s = s+'θ'
        if(result[0] == 30):
            s = s+'x'
        if(result[0] == 31):
            s = s+'y'
        if(result[0] == 32):
            s = s+'z'

        if(result[0] == 33):
            s = s+'π'
        if(result[0] == 34):
            s = s+'s'

        if(result[0] == 35):
            s = s+'√'

    print(s)

    flag1 = 0
    contain_trigo = True
    l = ['cos', 'sin', 'tan']
    l1 = ['+', '-', '*']
    for ch in l:
        if ch in s:
            s = s+'  :'+' Trigonometric Equation'
            print(s+'  :'+' Trigonometric Equation')
    if(flag):
        s = s+'     :'+' Summation'
        print(s)
    for ch in l1:
        if ch not in s and '=' not in s:
            contain_trigo = False
        else:
            contain_trigo = True

    if contain_trigo == False:
        for a in s:
            if a.isdigit() and '+' in s or '-' in s or '*' in s or '/' in s:
                flag1 = 0
            else:
                flag1 = flag1+1
        if flag1 == 0:
            ans = eval(s)
            s = s+'='+str(ans)+'  :'+' Arithmetic Expression'
            print(s)
        else:
            s = s+'  :'+' Arithmetic Expression'
            print(s)
    else:
        s = s+'  :'+' Arithmetic Expression'
        print(s)

    if '=' in s and '0' not in s and 'Arithmetic Expression' not in s:
        s = s+'  :'+' Equation of Line'
        print(s)
    if '=' in s and '0' in s and 'Arithmetic Expression' not in s:
        s = s+'  :'+' Linear Equation'
        print(s)

    string_list_for_OP.append(s)

    return render(request, 'Main/output.html', {'content': string_list_for_OP})
