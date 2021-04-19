import numpy as np
import time,os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model,Sequential
from keras.applications import ResNet101
from keras.layers.experimental.preprocessing import Rescaling
from keras.layers import Dense,Input,SeparableConv2D,Flatten,Dropout,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
from keras.datasets import mnist
# To disable all logging output from TensorFlow 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#function to plot Loss and accuracy Curves on training set
def plotgraph(history,value,val):
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.plot(history.history['accuracy'],'turquoise',linewidth=3.0)
  plt.legend(['Training loss','Training Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves and Accuracy Curves for {} {}'.format(value,val),fontsize=16)

#get_data function is used for fetching the training and testing from given directory for datasets
def get_data(path,value,num_classes,w):
  #the value passed in split to get training and testing is as per the requirement mentioned
  #for caltech101 dataset get 30 values form each class for training and remiaing for testing
  #fetch training data
  train=tf.keras.preprocessing.image_dataset_from_directory(
      directory=path,
      labels="inferred",
      label_mode="int",
      validation_split=value,
      subset="training",
      seed=123,
      image_size=(w,w),
      batch_size=1)
  
  #fetch testing data
  test=tf.keras.preprocessing.image_dataset_from_directory(
      directory= path,
      labels="inferred",
      label_mode="int",
      validation_split=value,
      subset="validation",
      seed=123,
      image_size=(w,w),
      batch_size=1)
  
  #noramlize the data 
  normalization = Rescaling(1./255)
  #normalize train and test data apply one hot encoding on labels and squeeze train and test to its required dimension
  #becomes equal to the shape of the input to the convolution network 
  train_n = train.map(lambda x, y: (tf.squeeze(normalization(x)), tf.squeeze(tf.one_hot(y,num_classes))))
  test_n = test.map(lambda x, y: (tf.squeeze(normalization(x)), tf.squeeze(tf.one_hot(y,num_classes))))
  
  #the output of train_n and test_n is mapped dataset so we iterate over the train_n and test_n
  #and store the features and labels in list
  train_inputs=[] #for training features
  train_labels=[] #for training labels
  test_inputs=[] #for testing features
  test_labels=[] #for testing labels
  c=0
  tr=iter(train_n) #create train_n iterator
  tt=iter(test_n)  #create test_n iterator
  for i in range(0,len(test_n)):
    if c<len(train_n):
      i,j=next(tr)
      train_inputs.append(i)
      train_labels.append(j)
      c+=1
    i,j=next(tt)
    test_inputs.append(i)
    test_labels.append(j)

  #convert the list to tensor
  train_inputs=tf.convert_to_tensor(train_inputs, dtype=tf.float32)
  train_labels=tf.convert_to_tensor(train_labels, dtype=tf.float32)
  test_inputs=tf.convert_to_tensor(test_inputs, dtype=tf.float32)
  test_labels=tf.convert_to_tensor(test_labels, dtype=tf.float32)
  
  #free memory
  del train_n,test_n,tr,tt,train,test
  return train_inputs,train_labels,test_inputs,test_labels

#design the model and get deep features
def design_model(trainx,testx,w,batch):
  image_input = Input(shape=(w,w,3))
  #using pretained weights from imagnet  
  model = ResNet101(input_tensor=image_input,include_top=False, weights='imagenet') 
  #get data from last block 
  last_layer = model.get_layer('conv5_block3_out').output
  #the ouput from last_layer contains too many features 
  x = GlobalAveragePooling2D()(last_layer)
  #generate the model 
  model= Model(image_input,x)
  #get deep features from the model
  train=model.predict(trainx,batch_size=batch,verbose=1,workers=50,
      use_multiprocessing=True)
  test=model.predict(testx,batch_size=batch,verbose=1,workers=50,
      use_multiprocessing=True)

  #model.summary()  
  #free memory 
  del model
  return train,test

#create own fully connected network as a classifer 
def deep_feature_classifier(trainx,testx,ytrain,ytest,nclass,epoch,batch,w,value):
  #list to store the training and testing accuracy
  tr_acc=[]
  test_acc=[]
  #tr and tt is used store training and testing time 
  tr=0.0
  tt=0.0
  #loop iterates three times to get get avergae accuracy for three runs for training and testing
  for i in range(0,3):
    #to calculate time for training
    st=time.time()
    #get deep features for every run  
    train,test=design_model(trainx,testx,w,batch)
    #my own fully connected classifier(ANN)
    classifier = Sequential([Dense(4096, activation='relu', input_shape=train[0].shape),
                         Dropout(0.5),
                         Dense(nclass, activation='softmax')])
    
    opt = Adam(lr=1e-4, decay=1e-4 / 50)
    classifier.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

    #reduce_lr method is used to reduce the learning rate if the learning rate is stagnant or if there are no major improvements during training
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
    #early stopping method is used to montior the loss if there are no significant reductions in loss then halts the training
    es = EarlyStopping(monitor='loss',patience=10)
    #fit the model
    history = classifier.fit(train,ytrain,epochs=epoch,
                          batch_size=batch,
                          shuffle=True,
                          verbose=1,
                          workers=50,
                          use_multiprocessing=True,  
                          callbacks=[reduce_lr,es])#lr_scheduler, 
    #plotgraph is used to plot the accuracy and loss graphs for three runs of training
    plotgraph(history,value,"{}".format('for run'+str(i+1)))
    
    #store the training time
    tr+=time.time()-st
    #append training accuracy
    tr_acc.append(history.history["accuracy"][-1:])
    acc=float(tr_acc[i][0])
    print("training accuracy for {} for run {} is : {}%".format(value,i+1,np.round(acc*100,2)))
    #testing start time
    st=time.time()
    
    #evalute the model
    (loss, accuracy) = classifier.evaluate(test,ytest, batch_size=batch,verbose=1,workers=50,
                          use_multiprocessing=True)
    
    print("testing accuracy for {} for run {} is : {}%".format(value,i+1,np.round(accuracy*100,2)))
    #store testing time
    tt+=time.time()-st
    #append testing accuracy
    test_acc.append(accuracy)

    #free memory
    del classifier,history,loss,accuracy,train,test,reduce_lr,es

  #rounding total time upto 4 decimal places for training and testing
  #rounding Average training and testing accuracy for 3 runs upto 2 decimal places
  print("Total training time for {} is {}".format(value,np.round(tr,4)))
  #calculate the average of training accuracy of 3 runs
  avg_tr=tf.add_n(tr_acc)/3
  avg_tr=np.float(avg_tr)
  print("Average of 3 runs training accuracy for {} is : {}%".format(value,np.round(avg_tr*100,2)))

  print("Total testing time for {} is {}".format(value,np.round(tt,4)))
  #calculate the average of testing accuracy of 3 runs
  avg_tt=tf.add_n(test_acc)/3
  avg_tt=np.float(avg_tt)
  print("Average of 3 runs testing accuracy for {} is : {}%".format(value,np.round(avg_tt*100,2)))

def mnist_data():
  (trainx, trainy), (testx, testy) = mnist.load_data()
  #normalize the data
  x_train_mean = np.mean(trainx, axis=(0,1,2))
  x_train_std = np.std(trainx, axis=(0,1,2))
  trainx = (trainx - x_train_mean) / x_train_std
  testx = (testx - x_train_mean) / x_train_std
  #one hot encoding
  trainy = tf.one_hot(trainy,10)
  testy = tf.one_hot(testy,10)
  # summarize loaded dataset
  print('Train Mnist: X=%s, y=%s' % (trainx.shape, trainy.shape))
  print('Test Mnist: X=%s, y=%s' % (testx.shape, testy.shape))
  #reshape mnist image
  trainx = trainx.reshape((trainx.shape[0], 28, 28, 1))
  testx = testx.reshape((testx.shape[0], 28, 28, 1))
  #convert grayscale image to rgb 
  trainx=tf.image.grayscale_to_rgb(tf.convert_to_tensor(trainx, dtype=tf.float32),name=None)
  testx=tf.image.grayscale_to_rgb(tf.convert_to_tensor(testx, dtype=tf.float32),name=None)
  #hyper parameters
  epoch=150 #epochs
  batch=32 #batch size
  nclass=10 #number of classes
  dim=28 #dim is used to initialize input dimension of the network
  #call the function to get the results
  deep_feature_classifier(trainx,testx,trainy,testy,nclass,epoch,batch,dim,"mnist")
  #free memory
  del trainx,testx,trainy,testy,x_train_mean,x_train_std

def caltech_data():
  #hyper parameters
  epoch=600 #epochs
  batch=20 #batch size
  nclass=102 #number of classes
  dim=224 #dim is used to initialize input dimension of the network
  #call scratch model,denset model with and without transfer learning for caltech101 dataset
  trainx,ytrain,testx,ytest=get_data('/content/drive/MyDrive/Caltech101',0.6654,nclass,dim)
  print("length of training data is {} and length of testing data is {}".format(len(trainx),len(testx)))
  #call the function to get the results
  deep_feature_classifier(trainx,testx,ytrain,ytest,nclass,epoch,batch,dim,"caltech101")
  #free memory
  del trainx,ytrain,testx,ytest

mnist_data()

caltech_data()

