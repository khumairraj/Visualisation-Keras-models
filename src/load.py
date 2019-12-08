'''
The main code files for loading the model and producing results to be studied
'''

from vis.utils import utils
from vis.visualization import visualize_cam
from deepexplain.tensorflow import DeepExplain
from vis.visualization import visualize_cam, overlay

import os
import json
import numpy as np
import urllib.request
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from skimage import feature, transform

import keras
import keras.backend as K
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.applications import MobileNetV2 as MOBILENET
from keras.models import model_from_json, model_from_yaml
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

import warnings
warnings.filterwarnings("ignore")

def loadclasslabels(source = 'imagenet'):
    '''
    Function to to load the labels. If the input is given as 'imagenet' then the imagenet labels are loaded.
    Else the return statement should be changed accordingly in the else clause. The output should be maintained in 
    all cases
    
    Parameters:
        source(string) : If it is 'imagenet', then imagenet labels are loaded. Else the else clause is modified.
    Returns:
        (list of strings) : The names of all the classes are returned into a list of string. The else clause must be 
        modified to do so.
    '''
    if(source == 'imagenet'):
        if not os.path.isfile(".//model//imagenet_class_index.json"):
            url = r'https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json'
            urllib.request.urlretrieve(url, ".//model//imagenet_class_index.json")
        CLASS_INDEX = json.load(open(".//model//imagenet_class_index.json"))
        classlabel  = []
        for i_dict in range(len(CLASS_INDEX)):
            classlabel.append(CLASS_INDEX[str(i_dict)][1])
        print("N of class={}".format(len(classlabel)))
        return classlabel
    else:
        #Here the code for the specific classlabel loading should be written
        CLASS_INDEX = json.load(open(source))
        classlabel  = []
        for i_dict in range(len(CLASS_INDEX)):
            classlabel.append(CLASS_INDEX[str(i_dict)][1])
        print("N of class={}".format(len(classlabel)))
        return classlabel


def loadmodel(modified = 0, source = 'imagenet', readtype = "plain"):
    '''
    Loads the model into the system. Also prepares a model by removing the last softmax layer to linear. This is important
    for GCAM to work.
    
    Parameters :
        source(string) : If it is 'imagenet', then the imagenet pretrained model is loaded. If the path is specified
                        then it is loaded from that path.
        modified(int) : If the model is modified before the previous run of the file then this is 1 else this is 0. This
                        saves time of changing the softmax layer.
        readtype(string) : Sets the readtype of the model.
                        
    Return :
        (keras model): The original model
        (keras model): The model with modified softmax to linear
    '''
    if source=='imagenet':
        model = MOBILENET(include_top = True)
        if (not os.path.isfile(".\\model\\tmp_model_save_rmsoftmax") or modified):
            temp = model.layers[-1].activation
            model.layers[-1].activation = keras.activations.linear
            model.save('.\\model\\tmp_model_save_rmsoftmax')
            modelwithlinear = keras.models.load_model('.\\model\\tmp_model_save_rmsoftmax')
            model.layers[layer_idx].activation = temp
        else:
            modelwithlinear = keras.models.load_model('.\\model\\tmp_model_save_rmsoftmax')
    else:
        #Different types of reading model.
        if(readtype == "json"):
            json_file = open(os.path.join(".\\model", 'model.json', 'r'))
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(".\\model", "model.h5"))
            print("Loaded model from disk")
        elif(readtype == "yaml"):
            yaml_file = open(os.path.join(".\\model", 'model.yaml', 'r'))
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            model = model_from_yaml(loaded_model_yaml)
            model.load_weights(os.path.join(".\\model", "model.h5"))
            print("Loaded model from disk")
        elif(readtype == "plain"):
            keras.models.load_model(os.path.join(".\\model", 'model'))

        if (not os.path.isfile(".\\model\\tmp_model_save_rmsoftmax") or modified):
            temp = model.layers[-1].activation
            model.layers[-1].activation = keras.activations.linear
            model.save('.\\model\\tmp_model_save_rmsoftmax')
            modelwithlinear = keras.models.load_model('.\\model\\tmp_model_save_rmsoftmax')
            model.layers[layer_idx].activation = temp
        else:
            modelwithlinear = keras.models.load_model('.\\model\\tmp_model_save_rmsoftmax')
    return model, modelwithlinear


def loaddata(imgfolderpath = ".\\sample"):
    '''
    Loads the images in the imgfolderpath to run the checks on the model.
    
    Parameters :
        imgfolderpath(string) : The path of the folder from where the image data has to be loaded.
        
    Returns:
        (np.array(batchsize, height, width, channels)) : The images to be fed into the model
        (list of images) : The list of original image to use while printing the results
        (list of string) : The names of the images to use while saving results
    
    '''
    imgsname = os.listdir(imgfolderpath)
    imagespath = [os.path.join(imgfolderpath, imgpath) for imgpath in imgsname]
    _imglist = []
    imglist = []
    for imgpath in imagespath:
        _img = load_img(imgpath,target_size=(224,224))
        img = img_to_array(_img)
        img = preprocess_input(img)
        _imglist.append(_img)
        imglist.append(img)
    imgs = np.array(imglist)
    return imgs, _imglist, imgsname