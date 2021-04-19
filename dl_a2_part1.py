import tensorflow as tf
from keras.layers.experimental.preprocessing import Rescaling,RandomFlip,RandomRotation
import os,time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Input,Dropout,BatchNormalization,Conv2D,MaxPooling2D,SeparableConv2D,add,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
# To disable all logging output from TensorFlow 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#get_data function is used for fetching the training and testing from given directory for both scene15 and caltech101 datasets
def get_data(path,value,num_classes,w):
  #the value passed in split to get training and testing is as per the requirement mentioned
  #for scene15 and caltech101 dataset so it gets randomly 100 images from class in 15scene for training and remaining for testing
  #similarly 30 values form each class in caltech101 for training and remiaing for testing
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
  
  return train_inputs,train_labels,test_inputs,test_labels

#function to plot Loss and accuracy Curves on training set for both the datasets
def plotgraph(history,value):
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.plot(history.history['accuracy'],'turquoise',linewidth=3.0)
  plt.legend(['Training loss','Training Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves and Accuracy Curves for {}'.format(value),fontsize=16)

#design_model function is used to create the model
def design_model(w,nclas):
  #generate data perform preporcessing on image 
  data = Sequential([RandomFlip("horizontal"),RandomRotation(0.1)])
  inputs = Input(shape=(w,w,3))
  #image processing 
  x = data(inputs)
  x = Conv2D(32, 3, strides=2, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  x = Conv2D(64, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  #as Vgg type of method of convolutions wont work for my dataset 
  #so i am creating my own custom layer architecture based on resnet methodology of using residual inputs
  #here i am using combination of separable convolutions plus conv2d
  #seperable convolutions first perform convolution which acts on each input channel separately 
  #followed by a pointwise convolution which mixes the resulting output channels.
  #this combination is useful as the size of training data is small for 15scene and caltech101 compared to the testing data 
  #keep the residual aside
  previous = x  
  #generate different size of the convolution layers for the model 
  for v in [128, 256, 512, 728]:
    x = Activation("relu")(x)
    x = SeparableConv2D(v, 3, padding="same")(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(v, 3, padding="same")(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding="same")(x)

    #projection of the residual output and add back the residual 
    residual = Conv2D(v, 1, strides=2, padding="same")(previous)
    x = add([x, residual])  
    #keep aside the residual for the next iteration
    previous = x  


  x = SeparableConv2D(1024, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  #at last before dense layer global average pooling is applied.
  #which applies average pooling on the spatial dimensions until each spatial dimension is one and leaves other dimensions unchanged. 
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  outputs = Dense(nclas,activation="softmax")(x)
  return Model(inputs,outputs)

#decay scheduler to set the different learning rates
def decay_schedule(epoch, lr):
  if (epoch % 10 == 0) and (epoch != 0):
    lr = lr * 0.1
  return lr

#scratch_model function is used to call the model, fit the model and evaluate and take average of three runs for training and testing
def scratch_model(x_train,y_train,x_test,y_test,nclas,epoch,batch,value,w):
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
    #call the model model is build each time for the three runs
    model = design_model(w,nclas)
    #this is called by the callback in model.fit method to schedule the decay of learning rates
    lr_scheduler = LearningRateScheduler(decay_schedule)
    #reduce_lr method is used to reduce the learning rate if the learning rate is stagnant or if there are no major improvements in training
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
    #early stopping method is used to montior the loss if there are no significant reductions in loss then it waits for max three iterations and then halts the training
    es = EarlyStopping(monitor='loss',patience=3)
    #adam is used as a optimizer
    ad = Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", optimizer=ad,
              metrics=["accuracy"])
    #fit the model
    history = model.fit(x_train,y_train,epochs=epoch,
                        batch_size=batch,
                        shuffle=True,
                        verbose=1,
                        workers=50,
                        use_multiprocessing=True,
                        callbacks=[lr_scheduler,reduce_lr,es])
    
    #plotgraph is used to plot the accuracy and loss graphs for three runs of training
    plotgraph(history,value)
    #store the training time
    tr+=time.time()-st
    #append training accuracy
    tr_acc.append(history.history["accuracy"][-1:])
    #testing start time
    st=time.time()
    #evalute the model
    (loss, accuracy) = model.evaluate(x_test,y_test, batch_size=batch, verbose=1)
    #store testing time
    tt+=time.time()-st
    #append testing accuracy
    test_acc.append(accuracy)
    
    #i delete due to the reduce space and as model is build again in every iteration so i delete the old part from memory 
    del model,history,loss,accuracy
  print("Training time for {} is {}".format(value,tr))
  #calculate the average of training accuracy of 3 runs
  avg_tr=tf.add_n(tr_acc)/3
  avg_tr=np.float(avg_tr)
  print("Average Training accuracy for {} is : {}%".format(value,avg_tr*100))

  print("Testing time for {} is {}".format(value,tt))
  #calculate the average of testing accuracy of 3 runs
  avg_tt=tf.add_n(test_acc)/3
  avg_tt=np.float(avg_tt)
  print("Average Testing accuracy for {} is : {}%".format(value,avg_tt*100))

#function to get the data and find training and testing accuracy for 15scene and caltech101 dataset
def main():
  #parameter for get_data
  #data path,split size,number of classes and dimension size
  #parameter passed in scratch_model for 15scene and caltech101 are
  #train features,train label,test features,test label,number of classes,epoch,batch size,dataset name and diemsnion size
  #function to get 15-scene data
  sx_train,sy_train,sx_test,sy_test=get_data('/content/drive/MyDrive/15-Scene',0.6655518394648829,15,180)
  print("length of training data is {} and length of testing data is {}".format(len(sx_train),len(sx_test)))
  #call the model
  scratch_model(sx_train,sy_train,sx_test,sy_test,15,60,32,"15-scene",180)
  #as i have limited memory i delete it 15 scene data after getting the results so there is space for caltech101 data
  del sx_train,sy_train,sx_test,sy_test
  #function to get caltech101 data
  cx_train,cy_train,cx_test,cy_test=get_data('/content/drive/MyDrive/Caltech101',0.6654,102,180)
  print("length of training data is {} and length of testing data is {}".format(len(cx_train),len(cx_test)))
  #call the model 
  scratch_model(cx_train,cy_train,cx_test,cy_test,102,60,32,"caltech101",180)
  del cx_train,cy_train,cx_test,cy_test

main()
