from google.colab import drive
drive.mount('/content/drive')
import os,time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization,Conv2D,Conv2DTranspose,Dense,Dropout,Flatten,Input,Reshape,UpSampling2D,ZeroPadding2D,LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam,SGD
from keras.applications import InceptionResNetV2,ResNet152V2,ResNet50,DenseNet201
from tensorflow import keras
from zipfile import ZipFile
from keras import backend as K
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.layers.experimental.preprocessing import Rescaling,RandomFlip,RandomRotation
from keras.layers import Activation,MaxPooling2D,SeparableConv2D,add,GlobalAveragePooling2D,Lambda
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
# To disable all logging output from TensorFlow 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
np.random.seed(1337)

#function to create the generator model
def create_generator(d,noise_dim,optimizer):
  generator = Sequential()  
  generator.add(Dense(d*d*256,kernel_initializer=RandomNormal(0,0.02),input_dim=noise_dim))
  generator.add(LeakyReLU(0.2))
  generator.add(Reshape((d,d,256)))
  generator.add(Conv2DTranspose(128,(4,4),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
  generator.add(LeakyReLU(0.2))  
  generator.add(Conv2DTranspose(128,(4,4),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
  generator.add(LeakyReLU(0.2)) 
  generator.add(Conv2DTranspose(128,(4,4),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
  generator.add(LeakyReLU(0.2))
  generator.add(Conv2D(3,(3,3),padding='same',activation='tanh',kernel_initializer=RandomNormal(0,0.02)))
  generator.compile(loss='binary_crossentropy', optimizer=optimizer)
  return generator

#function to create the discriminator module
def create_discriminator(w,optimizer):
  discriminator = Sequential()    
  discriminator.add(Conv2D(64,(3,3),padding='same',kernel_initializer=RandomNormal(0,0.02),input_shape=(w,w,3)))
  discriminator.add(LeakyReLU(0.2))  
  discriminator.add(Conv2D(128,(3,3),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Conv2D(128,(3,3),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
  discriminator.add(LeakyReLU(0.2))  
  discriminator.add(Conv2D(256,(3,3),strides=2,padding='same',kernel_initializer=RandomNormal(0,0.02)))
  discriminator.add(LeakyReLU(0.2))  
  discriminator.add(Flatten())
  discriminator.add(Dropout(0.4))
  discriminator.add(Dense(1,activation='sigmoid',input_shape=(w,w,3)))  
  discriminator.compile(loss='binary_crossentropy',optimizer=optimizer)
  return discriminator

def gan(train10X,Y_train10):
  #hyper parameters for GAN
  #size
  d=4
  #The dimension of noise
  noise_dim = 1
  batch_size = 1
  steps_per_epoch = 25
  epochs = 100
  w=32
  optimizer = Adam(0.0002, 0.5)

  #create empty list to store images
  new_images=[]
  new_labels=[]
  #create the generator and discriminator module
  discriminator = create_discriminator(w,optimizer)
  generator = create_generator(d,noise_dim,optimizer)

  # Make the discriminator untrainable when we are training the generator
  discriminator.trainable = False

  #link the two models to create the GAN
  gan_input = Input(shape=(noise_dim,))
  fake_image = generator(gan_input)

  gan_output = discriminator(fake_image)

  gan = Model(gan_input, gan_output)
  gan.compile(loss='binary_crossentropy', optimizer=optimizer)

  #constant noise for viewing how the GAN progresses
  static_noise = np.random.normal(0, 1, size=(1, noise_dim))
  print("epochs",epochs)
  print("steps_per_epoch",steps_per_epoch)
  #training loop
  for epoch in range(0,epochs):
    for batch in range(0,steps_per_epoch):
      #add noise
      noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
      #shuffle and get the real image from training set based on the batch szie every time
      real_x = train10X[np.random.randint(0, train10X.shape[0], size=batch_size)]
      #here i get the labels for the input images 
      real_y = Y_train10[np.random.randint(0, Y_train10.shape[0], size=batch_size)]
      #generate fake image
      fake_x = generator.predict(noise)
      x = np.concatenate((real_x, fake_x))
      #append newly generated images and labels in list
      new_images.extend(fake_x)
      new_labels.extend(real_y)
      disc_y = np.zeros(2*batch_size)
      disc_y[:batch_size] = 0.9
      #discriminator loss
      d_loss = discriminator.train_on_batch(x, disc_y)
      d_loss=np.round(d_loss,4) #limit output to 4 decimal places
      y_gen = np.ones(batch_size)
      #generator loss
      g_loss = gan.train_on_batch(noise, y_gen)
      g_loss=np.round(g_loss,4) #limit output to 4 decimal places
    im1 = tf.convert_to_tensor(real_x, tf.float32)
    im2 = tf.convert_to_tensor(fake_x, tf.float32)
    ssim_diff=tf.image.ssim(im1,im2,max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    avg_ssim=np.average(ssim_diff)
    avg_ssim="{:0.4f}".format(avg_ssim) #limit output to 4 decimal places
    print(f'Epoch: {epoch}') 
    print(f'SSIM Score: {ssim_diff}')
    print(f'Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss} \t\t Average SSIM score: {avg_ssim}')

    if epoch % 2 == 0:
        show_images(static_noise,epoch,generator,w)

  return new_images,new_labels

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
  train_n = train.map(lambda x, y: (tf.squeeze(normalization(x)),y))
  test_n = test.map(lambda x, y: (tf.squeeze(normalization(x)),tf.squeeze(tf.one_hot(y,num_classes))))
  
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

#function is used to get the gan data 
def get_gan_data(sx_train,sy_train,nclas):
  #plot the training dataset 
  plot_img(sx_train)
  #convert to numpy array
  ytrain=sy_train.numpy()
  sxtrain=sx_train.numpy()
  #call gan function
  st=time.time()
  new_images,new_labels=gan(sxtrain,ytrain)
  print("length of augumented images {} and labels {}".format(len(new_images),len(new_labels))) 
  print(f'Total time required: {np.round(time.time()-st,4)}')
  #merge original and newly generated images
  final_trainx=np.concatenate((sxtrain,new_images))
  final_trainy=np.concatenate((sy_train,new_labels))
  #one hot encode the labels
  final_trainy=np_utils.to_categorical(final_trainy, num_classes=nclas, dtype='float32')
  print("length of final images {} and labels {}".format(len(final_trainx),len(final_trainy))) 
  return final_trainx,final_trainy

#function to plot 5 images from the training dataset
def plot_img(train10X):
  for i in range(5):
    # define subplot
	  plt.subplot(7, 7, 1 + i)
	  # turn off axis
	  plt.axis('off')
	  # plot raw pixel data
	  plt.imshow(train10X[i])
  plt.show()

#function to display images generated by generator
def show_images(noise,epoch,generator,w):
  channels=3
  generated_images = generator.predict(noise)
  plt.figure(figsize=(10,10))
  for i, image in enumerate(generated_images):
    plt.subplot(10,10,i+1)
    if channels == 1:
      plt.imshow(np.clip(image.reshape((w,w)),0.0,1.0),cmap='gray')
    else:
      plt.imshow(np.clip(image.reshape((w,w,channels)),0.0,1.0))
    plt.axis('off')
    
  plt.tight_layout()

#function to plot Loss and accuracy Curves on training set for both the datasets
def plotgraph(history,value):
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.plot(history.history['accuracy'],'turquoise',linewidth=3.0)
  plt.legend(['Training loss','Training Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves and Accuracy Curves for {}'.format(value),fontsize=16)

#scratch_model function is used to call the model, fit the model and evaluate and take average of three runs for training and testing
def scratch_model(x_train,y_train,x_test,y_test,nclas,epoch,batch,value,w):
  #tr and tt is used store training and testing time 
  tr=0.0
  tt=0.0
  #to calculate time for training
  st=time.time()
  #input image 
  image_input = Input(shape=(None,None,3))
  #lambda layer is added to make sure images are resized to 224,224,3 
  prep=Lambda(lambda x: tf.image.resize(x,(224, 224)))(image_input)
  #inceptionresnetv2 model
  dmodel = InceptionResNetV2(include_top=False, weights=None,input_shape=(224,224,3))(prep)
  x = GlobalAveragePooling2D()(dmodel)
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x) 
  #adding my own output classifier
  out = Dense(nclas, activation='softmax', name='output')(x)
  model = Model(image_input, out)

  #reduce_lr method is used to reduce the learning rate if the learning rate is stagnant or if there are no major improvements in training
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
  #early stopping method is used to montior the loss if there are no significant reductions in loss then it halts the training
  es = EarlyStopping(monitor='loss',patience=20)
  #adam is used as a optimizer
  opt = Adam(lr=1e-4, decay=1e-4 / 50)
  model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
  #fit the model
  history = model.fit(x_train,y_train,epochs=epoch,
                        batch_size=batch,
                        validation_data=((x_test,y_test)),
                        validation_batch_size=batch,
                        shuffle=True,
                        verbose=1,
                        workers=50,
                        use_multiprocessing=True,
                        callbacks=[reduce_lr,es])
    
  #plotgraph is used to plot the accuracy and loss graphs for three runs of training
  plotgraph(history,value)
  #store the training time
  #training accuracy
  tr_acc=history.history["accuracy"][-1:]
  tracc=float(tr_acc[0])
  #testing accuracy
  tt_acc=history.history["val_accuracy"][-1:]
  ttacc=float(tt_acc[0])
  print("training accuracy for {} is : {}%".format(value,np.round(tracc*100,2)))
  print("testing accuracy for {} is : {}%".format(value,np.round(ttacc*100,2)))
  tr=time.time()-st 
  print("Total time required for {} is {}".format(value,np.round(tr,4)))

#function to get the data and apply gan and write train and test set values in a numpy file
def scene15data():
  #hyperparameters for scene15
  epochs=500
  size=32
  w=224
  nclas=15

  #parameter for get_data
  #data path,split size,number of classes and dimension size
  #parameter passed in scratch_model for 15scene and caltech101 are
  #train features,train label,test features,test label,number of classes,epoch,batch size,dataset name and diemsnion size
  #function to get 15-scene data
  sx_train,sy_train,sx_test,sy_test=get_data('/content/drive/MyDrive/15-Scene',0.6655518394648829,15,size)
  print("length of training data is {} and length of testing data is {}".format(len(sx_train),len(sx_test)))
  #call function
  final_trainx,final_trainy=get_gan_data(sx_train,sy_train,nclas)
  #write to npy file
  np.save('/content/drive/MyDrive/sxtrain.npy',final_trainx)
  np.save('/content/drive/MyDrive/sytrain.npy',final_trainy)
  np.save('/content/drive/MyDrive/sxtest.npy',sx_test)
  np.save('/content/drive/MyDrive/sytest.npy',sy_test)
  
#function to get the data and apply gan and write train and test set values in a numpy file
def caltech101data():
  #hyperparameters for caltec101
  epochs=500
  size=32
  w=224
  nclas=102
  #function to get caltech101 data
  cx_train,cy_train,cx_test,cy_test=get_data('/content/drive/MyDrive/Caltech101',0.6654,nclas,size)
  print("length of training data is {} and length of testing data is {}".format(len(cx_train),len(cx_test)))
  
  #call the function
  final_trainx,final_trainy=get_gan_data(cx_train,cy_train,nclas)
  #write to npy file
  np.save('/content/drive/MyDrive/cxtrain.npy',final_trainx)
  np.save('/content/drive/MyDrive/cytrain.npy',final_trainy)
  np.save('/content/drive/MyDrive/cxtest.npy',cx_test)
  np.save('/content/drive/MyDrive/cytest.npy',cy_test)

scene15data()

caltech101data()

#load scene15 data and call the scratch model 
def load_scene():
  #load data from npy file
  final_trainx=np.load('/content/drive/MyDrive/sxtrain.npy')
  final_trainy=np.load('/content/drive/MyDrive/sytrain.npy')
  sx_test=np.load('/content/drive/MyDrive/sxtest.npy')
  sy_test=np.load('/content/drive/MyDrive/sytest.npy')
  print("length of train {} and test {} after GAN agumentation".format(len(final_trainx),len(sx_test)))
  #call the function
  scratch_model(final_trainx,final_trainy,sx_test,sy_test,15,100,128,"15-scene",224)

#load caltech101 data and call the scratch model 
def load_caltech():
  #load data from npy file
  final_trainx=np.load('/content/drive/MyDrive/cxtrain.npy')
  final_trainy=np.load('/content/drive/MyDrive/cytrain.npy')
  cx_test=np.load('/content/drive/MyDrive/cxtest.npy')
  cy_test=np.load('/content/drive/MyDrive/cytest.npy')
  print("length of train {} and test {} after GAN agumentation".format(len(final_trainx),len(cx_test)))
  #call the function
  scratch_model(final_trainx,final_trainy,cx_test,cy_test,102,100,128,"Caltech-101",224)

load_scene()

load_caltech()

#LSTM Time Series Model

#plot the output of timeseries
def plot_output(history,value,original,predicted):
  #plot the loss curves for training
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.legend(['Training loss'],fontsize=18)
  plt.xlabel('Epochs',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves for {}'.format(value),fontsize=16)
  plt.show()
  #plot the original and predicted values of wheather forecast for temperature
  plt.figure(figsize=[20,8])
  plt.plot(original,color='firebrick',label='original',linewidth=1,marker='o')
  plt.plot(predicted,color='turquoise',label='predicted',linewidth=1,marker='x')
  plt.legend()
  plt.show()

#function to normalize the dataset
def normalize(data, train_split):
  data_mean = data[:train_split].mean(axis=0)
  data_std = data[:train_split].std(axis=0)
  return (data - data_mean) / data_std

#function to calculate RMSE error as loss for keras model.fit and model.evaluate
def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true))) 

#function to get the wheather forecast data from the given link
def get_climate_data():
  url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
  zip_path = keras.utils.get_file(origin=url, fname="jena_climate_2009_2016.csv.zip")
  zip_file = ZipFile(zip_path)
  zip_file.extractall()
  csv_path = "jena_climate_2009_2016.csv"
  df = pd.read_csv(csv_path)
  return df

#function to preprocess the dataset and generate train and test sets
def climate_preporcess(df,train_split,step,past,future,batch_size):
  #normalize the data
  features = normalize(df[df.columns[1:]], train_split)
  #split data into training and testing
  train_data = features.loc[0 : train_split - 1]
  test_data = features.loc[train_split:]
  #set the start and end for splitting on training
  start = past + future
  end = start + train_split
  #get the input features and labels for training
  x_train = train_data.values
  y_train = features["T (degC)"].iloc[start:end]
  #set the sequence length
  sequence_length = int(past / step)
  #The timeseries_dataset_from_array function takes in a sequence of data-points gathered at equal intervals, 
  #along with time series parameters such as length of the sequences/windows, spacing between two sequence/windows. 
  #to produce batches of sub-timeseries inputs and targets sampled from the main timeseries.
  train = keras.preprocessing.timeseries_dataset_from_array(x_train,y_train,sequence_length=sequence_length,
                                                            sampling_rate=step,batch_size=batch_size)

  #set the split for testing
  x_end = len(test_data) - past - future
  label_start = train_split + past + future
  #get the input feeatures and labels for testing
  x_test = test_data.iloc[:x_end].values
  y_test = features["T (degC)"].iloc[label_start:]

  #The validation dataset must not contain the last 792 rows as we won't have label data for those records, 
  #hence 792 must be subtracted from the end of the data.
  #The validation label dataset must start from 792 after train_split, hence we must add past + future (792) to label_start.
  test = keras.preprocessing.timeseries_dataset_from_array(x_test,y_test,sequence_length=sequence_length,
                                                           sampling_rate=step,batch_size=batch_size)

  return train,test

#function time series model using lstm
def timeseries_model(train,test,epoch):
  #start time
  st=time.time()
  #print the shape of the input features
  for batch in train.take(1):
    inputs, targets = batch

  print("Input shape:", inputs.numpy().shape)
  print("Target shape:", targets.numpy().shape)
  
  #LSTM time series model
  inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
  lstm_out = LSTM(64)(inputs)
  outputs = Dense(1)(lstm_out)
  model = keras.Model(inputs=inputs, outputs=outputs)

  #reduce_lr method is used to reduce the learning rate if the learning rate is stagnant or if there are no major improvements in training
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                patience=5, min_lr=0.001)

  #early stopping condition
  es = EarlyStopping(monitor="loss", mode='min', verbose=1,patience=5)
  
  #compile model
  model.compile(optimizer=SGD(learning_rate=0.001,decay=0.001/25),loss=root_mean_squared_error)
  model.summary()
  
  #fit the model
  history = model.fit(train,epochs=epoch,callbacks=[reduce_lr,es],
                      verbose=1,workers=50,
                      use_multiprocessing=True)
  
  print(f'Total time required for training: {np.round(time.time()-st,4)}')
  #get the loss
  tr=history.history["loss"][-1:]

  #start time 
  st=time.time()
  #predict the model
  predicted = model.predict(test,
                            verbose=1,workers=50,
                            use_multiprocessing=True)
  
  print(f'Total time required for testing: {np.round(time.time()-st,4)}')
  #list to store original and predicted values
  original=[]
  predicted=[]
  for x, y in test.take(len(test)):
    original.append(y[0].numpy())
    predicted.append(model.predict(x)[0])

  #claculate testing loss
  rmse = sqrt(mean_squared_error(original,predicted))

  #plot the graph for training loss and predicted and original value
  plot_output(history,"wheather forecast of temperature",original,predicted)

  print("Training loss for time series model on wheather forecast is : {}".format(np.round(tr[0],4)))
  print("Testing loss for time series model on wheather forecast is : {}".format(np.round(rmse,4)))

#main function to call the above function in a sequence
def main():
  #get the data
  df=get_climate_data()
  #plot the different columns of dataframe
  df.plot(use_index="datetime",figsize=(20,10),subplots=True,title=["Pressure","Temperature","Temperature in Kelvin",
                                                                 "Temperature (dew point)","Relative Humidity",
                                                                 "Saturation vapor pressure","Vapor pressure",
                                                                 "Vapor pressure deficit","Specific humidity",
                                                                 "Water vapor concentration","Airtight","Wind speed",
                                                                 "Maximum wind speed","Wind direction in degrees"],legend=True)
  
  #hyper parameters
  #split data into 70% training and 30% testing
  train_split = int(0.7 * int(df.shape[0]))
  #tracking data from past 720 timestamps that is (720/6=120 hours). 
  #This data will be used to predict the temperature after 72 timestamps (76/6=12 hours).
  #The model is shown data for first 5 days i.e. 720 observations, that are sampled every hour. 
  #The temperature after 72 (12 hours * 6 observation per hour) observation will be used as a label.
  step = 6
  past = 720
  future = 72
  batch_size = 256
  epoch = 100

  #call the preprocess function and get train and test data generated
  train,test=climate_preporcess(df,train_split,step,past,future,batch_size)
  print("length of training {} and testing {}".format(len(train),len(test)))
  #call the time series model
  timeseries_model(train,test,epoch)

main()

