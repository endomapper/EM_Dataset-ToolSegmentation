import os
import tensorflow as tf
import nets.MiniNetModif as MiniNetModif
import numpy as np
from utils_mininet.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, apply_augmentation, get_metrics,init_model
import glob
from PIL import Image
import cv2

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """

    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def create_clasifier(base_model):
    base_model.trainable = False
    input_shape = (1056,1280,3)
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs,outputs)
    restore_state(model,'weights_mininet/clasif_mininet_endomapper/model1_0')
    model_encod = tf.keras.Model(inputs = inputs, outputs = model.layers[-4].output)
    input_2 = tf.keras.Input(shape=(132, 160, 128))
    y = input_2
    for layer in model.layers[2:]:
        y = layer(y)
    output_2 = y
    model_class = tf.keras.Model(inputs = input_2, outputs = output_2)
    return model_encod, model_class

def main():
    print(tf.__version__)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_encod = MiniNetModif.MiniNetv2(num_classes=2,include_top=True)
    model_decod = MiniNetModif.MiniNetDecod(num_classes=2,include_top=True)

    restore_state(model_decod, 'weights_mininet/mininet_endomapper/model_0')

    model_encod, model_class = create_clasifier(model_encod)

    files = [glob.glob('data/images/*.png')]

    for i in range(len(files)):
        files[i].sort()

        for j in range(len(files[i])):
            img_path = files[i][j]
            
            img = tf.keras.preprocessing.image.load_img(img_path,0)
            img0 = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
            img1 = img0/127.5 - 1 
            input_img = np.expand_dims(img1,0)

            if (input_img.shape == (1,1080,1440,3)):
                image = input_img[:,24:,160:,:]
            elif (input_img.shape == (1,1056,1920,3)):
                image = input_img[:,:,640:,:]
            else:
                image = input_img
            

            features = model_encod(image,training=False)
            
            tool = model_class(features,training = False)
           
            tool = np.array(tool>0.5).astype(np.uint8)
            tool = int(tool)
            
            if tool == 1:
                mask_filt = model_decod(image, features, training = False)
                mask_filt = np.asarray(mask_filt)
                mask_filt = mask_filt.squeeze(0)
                mask_filt = mask_filt[:,:,1]
            else : 
                mask_filt = np.zeros((image.shape[1],image.shape[2]))
            
            mask_filt = np.uint8(mask_filt>0.5)*255
            
            res_mask = Image.fromarray(mask_filt)
            res_mask.save('results/'+img_path.split('/')[-1])
            
            res_overlay = img0.astype(np.uint8)
            res_overlay[24:,160:,:] = mask_overlay(res_overlay[24:,160:,:], mask_filt,color=(0,1,0))
            res_overlay = Image.fromarray(res_overlay)
            res_overlay.save('results_overlay/'+ img_path.split('/')[-1])
            

if __name__ == '__main__':
    main()

