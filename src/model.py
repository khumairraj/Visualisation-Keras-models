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

def runmodel(model, imgs):
    '''
    Runs the model on the preprocessed data.
    
    Parameters:
        model(keras model): The model which will run to produce the result to be analysed
        imgs(np.array(batchsize, height, width, channels)) : The preprocesses images to be fed in the model.
        
    Results:
        (np.array(batchsize, numberofclass)) : The softmax output of the model
        (list of list of int) : The predicted class of the image in a sorted manner.
    '''
    preds = model.predict(imgs)
    class_idxs_sortedlist = []
    for pred in preds:
        class_idxs_sorted = np.argsort(pred.flatten())[::-1]
        class_idxs_sortedlist.append(class_idxs_sorted)
    return preds, class_idxs_sortedlist

def printtopnclass(class_idxs_sorted, topNclass = 5):
	'''
	Prints the class output
	'''
	for i, idx in enumerate(class_idxs_sorted[:topNclass]):
		print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(i + 1,classlabel[idx],idx,y_pred[0,idx]))

def plot_custom(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = -abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='bicubic', cmap=cmap, vmin=abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='bicubic', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis