'''
Codefile to produce the results for the following types: GCAM
'''

from vis.utils import utils
from vis.visualization import visualize_cam
from deepexplain.tensorflow import DeepExplain
from vis.visualization import visualize_cam, overlay

import os
import sys
import json
import tqdm
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
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

import warnings
warnings.filterwarnings("ignore")

sys.path.append(".\\src")
from model import *
from load import *

with open('params.json') as json_file:
    params = json.load(json_file)

if not os.path.exists(params["savepath"]):
    os.mkdir(params["savepath"])
savepath = params["savepath"]

classlabel = loadclasslabels(source = params['sourceclasslabel'])
model, modelwithlinear = loadmodel(modified = params['whethermodelmodified'], source = params['sourcemodel'], readtype = params["readtypeofmodel"])
imgs, _imglist, imgsname = loaddata(imgfolderpath = params['imgfolderpath'])
preds, class_idxs_sortedlist = runmodel(model, imgs)

topNclass = params["topNclass"]
layer_idx = utils.find_layer_idx(model, 'Logits')
for penultimate_layer_name in tqdm.tqdm(params["penultimate_layer_names"]):
    penultimate_layer_idx = utils.find_layer_idx(model, penultimate_layer_name) 

    for u, img in tqdm.tqdm(enumerate(imgs)):
        class_idxs_sorted = class_idxs_sortedlist[u]
        _img = _imglist[u]
        img = seed_input = imgs[u]
        y_pred = preds[u]
        imgname = imgsname[u]

        labelstostudy = class_idxs_sorted[:topNclass]
        grads_list = []
        class_idx_list = []
        fig, axes = plt.subplots(1, 1+ len(labelstostudy),figsize=(7*(1 + len(labelstostudy)),5))
        axes[0].imshow(_img)

        for j, class_idx in enumerate(labelstostudy):
            grads  = visualize_cam(modelwithlinear,layer_idx,class_idx, seed_input,
                                   penultimate_layer_idx = penultimate_layer_idx,
                                   backprop_modifier     = None,
                                   grad_modifier         = None)
            axes[j+1].imshow(_img)
            axes[j+1].imshow(grads,cmap="jet",alpha=0.8)
            axes[j+1].set_title("Pr(class={}) = {:5.2f}".format(
                              classlabel[class_idx],
                              y_pred[class_idx]))   
        plt.savefig(os.path.join(savepath, imgname.split(".")[-2] + "_" + penultimate_layer_name + ".jpg"))