#mount drive to use files 
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import time,os,pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense,Activation,Flatten,Input
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16
# To disable all logging output from TensorFlow 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#this function is used to preprocess 15scene dataset
#it takes path of file then extracts images resize them to 224,224,3 
#converts the image to array encode the label with one hot encoder
#then all the images are stored in a one data frame 
#then randomly 100 images from each calss are selected as train set and remaining are testing set
#del command of python is used to free memory occupied by variables
def preprocess_data(path):
  df=[]
  data_dir=os.listdir(path)
  for i in data_dir:
    data=os.listdir('{}/{}'.format(path,i))
    for j in data:
      #load image and resize
      image = load_img('{}/{}/{}'.format(path,i,j), target_size=(224, 224))
      #image to array
      x = img_to_array(image)
      #reshape to 224,224,3  
      x = x.reshape(x.shape) 
      df.append([x,i])
  final=pd.DataFrame(df)
  final.columns=["data","labels"]
  #randomly selecting 100 images for training and remaining for testing
  train = final.groupby('labels').apply(lambda x:x.sample(100)).reset_index(drop=True)[["data","labels"]]
  #print("count of classes\n",train["labels"].value_counts())
  test=final.loc[final.index.difference(train.index), ]
  #print("length of train is {} and test is {}".format(len(train),len(test)))
  train.reset_index(drop=True,inplace=True)
  test.reset_index(drop=True,inplace=True)
  #normalize
  train["data"]=train["data"]/255
  test["data"]=test["data"]/255
  #convert to tensor
  strain=tf.constant(train["data"].to_list())
  stest=tf.constant(test["data"].to_list())
  #one hot encoding
  Y_train = np_utils.to_categorical(train["labels"],15)
  Y_test = np_utils.to_categorical(test["labels"],15)

  #delete variables and free memory
  del df,x,final,path,data,image,data_dir,train,test
  return strain,stest,Y_train,Y_test

#function to plot Loss and accuracy Curves on training set
def plotgraph(history,value):
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.plot(history.history['accuracy'],'turquoise',linewidth=3.0)
  plt.legend(['Training loss','Training Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves and Accuracy Curves for {}'.format(value),fontsize=16)


#preprocess data according to VGG16 format caltech101 datasets
def preprocess(path):
  train=[]
  tr=[]
  test=[]
  tt=[]
  data_dir=os.listdir(path)
  for i in data_dir:
    data=os.listdir('{}/{}'.format(path,i))
    c=0
    for j in data:
      #load image and resize to 224,224
      image = load_img('{}/{}/{}'.format(path,i,j), target_size=(224, 224))
      #convert image to array
      x = img_to_array(image)  
      #reshape to 224,224,3
      x = x.reshape(x.shape)
      #normalize
      x=x/255
      #one hot encode
      y = np_utils.to_categorical(i,103)
      #select 30 values from each class as training and remaining are used for testing
      if c<30:
        train.append(x)
        tr.append(y)
        c+=1
      else:
        test.append(x)
        tt.append(y)

  train=np.array(train)
  test=np.array(test)
  tr=np.array(tr)
  tt=np.array(tt)
  #del the varaibles
  del x,data,image,data_dir,y
  return train,test,tr,tt

#with transfer learning for 15sene and caltech dataset
def Vgg_transfer(xtrain,xtest,Y_train,Y_test,value,epoch,tr_batch,tt_batch,nclass):
    for i in range(0,3):
        #start time training
        st=time.time()
    
        # input shape for vgg16 model
        image_input = Input(shape=(224, 224, 3))
        #using pretained weights from imagnet for transfer learning 
        model = VGG16(input_tensor=image_input,include_top=True, weights='imagenet')  
        last_layer = model.get_layer('block5_pool').output
        x= Flatten(name='flatten')(last_layer)
        #keeping fc1 and fc2 layers of vgg16 to default value only which is 4096 and adding my own classifier layer at the end
        x = Dense(4096, activation='relu', name='fc1')(x) 
        x = Dense(4096, activation='relu', name='fc2')(x) 
        #adding own classifer layer for output, nclass denotes number of classes 
        out = Dense(nclass, activation='softmax', name='output')(x)
        #creating my model with  my own classifier layer at last and keeping all above layers to their default value 
        custom_vgg_model2 = Model(image_input, out)

        del model,last_layer,image_input,x,out
        
        # Compile the model, i am using RMSprop optimizer
        optim = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
        custom_vgg_model2.compile(loss='categorical_crossentropy',
                                  optimizer=optim,
                                  metrics=['accuracy'])

       #list to store the training and testing accuracy
       tr_acc=[]
       test_acc=[]
       #tr and tt is used store training and testing time 
       tr=0.0
       tt=0.0

       history = custom_vgg_model2.fit(xtrain,Y_train,
                                       epochs=epoch,
                                       batch_size=tr_batch,
                                       shuffle=True,
                                       verbose=2)
    
       tr+=time.time()-st
       #plot accuracy and loss curve 
       plotgraph(history,value)
       tr_acc.append(history.history["accuracy"][-1:])
       st=time.time()
       #evalute the model
       (loss, accuracy) = custom_vgg_model2.evaluate(xtest, Y_test, batch_size=tt_batch, verbose=2)
       tt+=time.time()-st
       test_acc.append(accuracy)
     
  print("Training time with transfer learning for {} is {}".format(value,tr))
  #calculate the average of training accuracy of 3 runs
  avg_tr=tf.add_n(tr_acc)/3
  avg_tr=np.float(avg_tr)
  print("Training accuracy with transfer learning for {} is : {}%".format(value,avg_tr*100))

  print("Testing time with transfer learning for {} is {}".format(value,tt))
  #calculate the average of testing accuracy of 3 runs
  avg_tt=tf.add_n(test_acc)/3
  avg_tt=np.float(avg_tt)
  print("Testing accuracy with transfer learning for {} is : {}%".format(value,avg_tt*100))
  del tr_acc,avg_tr,avg_tt,test_acc,custom_vgg_model2,optim,tt,tr

#learning from scratch on 15 scene and caltech101 datasets, here i dont use any pretrained weights so model learns from scratch
def Vgg_normal(xtrain,xtest,Y_train,Y_test,value,epoch,tr_batch,tt_batch,nclass):   
  #imput shape for the model
  image_input = Input(shape=(224, 224, 3))
  #creating model from scratch if i pass input weights in this then it becomes pretrained else it is becomes learning from scratch
  #so model is built from scratch 
  model = VGG16(input_tensor=image_input,include_top=True)
  last_layer = model.get_layer('block5_pool').output
  x= Flatten(name='flatten')(last_layer)
  x = Dense(4096, activation='relu', name='fc1')(x) 
  x = Dense(4096, activation='relu', name='fc2')(x) 
  #adding my own output classifier
  out = Dense(nclass, activation='softmax', name='output')(x)
  custom_vgg_model2 = Model(image_input, out)
  del model,last_layer,image_input,x,out

  #using RMSprop optimizer and compileing my custom model
  optim = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
  custom_vgg_model2.compile(loss='categorical_crossentropy',
                optimizer=optim,
                metrics=['accuracy'])

  #start time
  st=time.time()
  
  history = custom_vgg_model2.fit(xtrain,Y_train,
                                    epochs=epoch,
                                    batch_size=tr_batch,
                                    shuffle=True,
                                    verbose=2)

  acc=history.history["accuracy"][-1:]
  acc=np.float(acc[-1])
  #plot accuracy and loss graph
  plotgraph(history,value)
  print("Training time without transfer learning for {} is {}".format(value,time.time()-st))
  print("Training accuracy without transfer learning for {} is : {}%".format(value,acc*100))
  st=time.time()
  (loss, accuracy) = custom_vgg_model2.evaluate(xtest, Y_test, batch_size=tt_batch, verbose=2)
  print("Testing time without transfer learning for {} is {}".format(value,time.time()-st))
  print("Testing accuracy without transfer learning for {} is : {}%".format(value,accuracy*100))
  del custom_vgg_model2,optim,st

#this function loads 15scene dataset and call method Vgg_transfer for transfer learning and Vgg_normal for learning from scratch
def scene():
  #provide path details here for 15scene
  strain,stest,Y_train,Y_test=preprocess_data('/content/drive/MyDrive/15-Scene')
  #hyperparameter
  epoch=5
  tr_batch=1 
  tt_batch=1
  Vgg_transfer(strain,stest,Y_train,Y_test,"15Scene",epoch,tr_batch,tt_batch,15)
  print("length of train is {} and test is {}".format(len(strain),len(stest)))
  
  #hyperparameter
  epoch=15
  tr_batch=1
  tt_batch=1 
  Vgg_normal(strain,stest,Y_train,Y_test,"15Scene",epoch,tr_batch,tt_batch,15)
    
  del strain,stest,epoch,tr_batch,tt_batch
  

#this function loads caltech101 dataset and call method Vgg_transfer for transfer learning and Vgg_normal for learning from scratch
def caltech():
  #proivide path details here for caltech101
  strain,stest,Y_train,Y_test=preprocess('/content/drive/MyDrive/Caltech101')
  print("length of train is {} and test is {}".format(len(strain),len(stest)))
  #hyperparameter
  epoch=3
  tr_batch=1
  tt_batch=1
  Vgg_transfer(strain,stest,Y_train,Y_test,"caltech101",epoch,tr_batch,tt_batch,103)
  
  #hyperparameter
  epoch=10
  tr_batch=1
  tt_batch=1
  Vgg_normal(strain,stest,Y_train,Y_test,"caltech101",epoch,tr_batch,tt_batch,103)
    
  del strain,stest,epoch,tr_batch,tt_batch

scene()
caltech()


