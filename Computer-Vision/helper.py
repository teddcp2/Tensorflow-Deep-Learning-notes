# File which contains helper method to visualize, process, predict for CNN models
# Dt- 02.08.21

import os
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf


#####3
## Method 1
def get_class_names(path):
    '''
    getting all the classNames from the directory structure
    '''
    
    if not path :
        raise Exception("Please provide path as an arguement")
    
    dir= pathlib.Path(path)
    classes= []
    
    for item in dir.glob("*") :
        classes.append(item.name)
    
    classes= np.sort(np.array(classes))
    classes= classes[classes != ".DS_Store"]
    
    return classes

#####
## Method 2
def traverse_dataset(path):
    '''
    A method which will traverse the direcotry and count the files and type
    '''
    
    if not path :
        raise Exception("Please provide path as an arguement")
    
    for dir_path, dir_names, file_names in os.walk(path):
        print(f" There are {len(file_names)} files in {len(dir_names)} direcotory under {dir_path}")
    
    # os.listdir(path) will gnerate the list of files on given path..

#######
# Method 3
def view_random_image(target_dir, target_class) :
    """
    getting a random image from a folder under a particular class 
    """
    target= os.path.join(target_dir, target_class)
    img= random.sample(os.listdir(target),1)[0]
    path= os.path.join(target, img)
    
    img_tensor=mpimg.imread(path)
    plt.imshow(img_tensor)
    plt.axis(False)
    plt.title(target_class)
    
    print("shape --", img_tensor.shape)
    return img_tensor

#######
## Method 4
def plot_curves(hist) :
    '''
    Plotting the loss and accuracy curves
    '''
    
    loss= hist.history["loss"]
    val_loss= hist.history["val_loss"]
    acc= hist.history["accuracy"]
    val_acc= hist.history["val_accuracy"]
    
    epochs= range(len(loss))
    
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, val_loss, label="val-loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, acc, label="Accuracy")
    plt.plot(epochs, val_acc, label="val-accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.tight_layout()
    
    
def load_prep_image(file, shape=224):
    '''
    Reads an image file and loads with scaling
    '''
    
    img= tf.io.read_file(file)
    
    img_tensor= tf.image.decode_image(img, channels=3),
    
    img_resized= tf.image.resize(img_tensor, size=(shape, shape))
    
    img_resized /= 255.
    
    return img_resized

def pred_and_plot(model,  file, classes) :
    """
    Predicting and plotting for an image
    """
    img= load_prep_image(file)
    
    if img.ndim < 4:
        plt.imshow(img)
        img= tf.expand_dims(img, axis=0) 
    else:
        plt.imshow(tf.squeeze(img, axis=0))

    pred= tf.squeeze(model.predict(img))
    print(pred)
    if len(pred) ==1 :
        pred_Class= classes[tf.cast(pred.numpy() > 0.5 , dtype="int8").numpy()]
    else :
        pred_Class= classes[pred.argmax()]
    
    plt.title(pred_Class, fontdict={"fontsize": 18, "color": "green"})
    plt.axis(False)
    
        

if __name__ == "__main__":
    pass