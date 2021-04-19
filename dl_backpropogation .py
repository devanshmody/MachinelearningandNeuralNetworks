import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

#function to preprocess data according to format required for traning
#preprocessing of data for classification dataset
def preprocessing_c():
  #read the data files into dataframe
  data1=pd.read_csv("/content/drive/MyDrive/datatraining.txt")
  data2=pd.read_csv("/content/drive/MyDrive/datatest.txt")
  data3=pd.read_csv("/content/drive/MyDrive/datatest2.txt")

  #cobime all dataframe itno one dataframe of 20560 rows
  final=[data1,data2,data3]
  df=pd.concat(final)

  #scale the numeric values using min max scaler to normalize and get the target output
  t=1
  list1=["Temperature","Humidity","Light","CO2"]
  for i in list1:
    df[i]=(df[i]-df[i].mean())/(df[i].max()-df[i].mean())
    t*=df[i]

  df["target"]=t
  df["target"]=np.sign(df["target"]) #ifnegative -1 if positve 1 else 0
  df=df[["Temperature","Humidity","Light","CO2","target"]].round(2)

  #shuffle dataframe and split dataframe in 70% train and 30% test
  df= df.sample(frac=1).reset_index(drop=True) 
  train_size = int(0.7 * len(df)) 
  train_set = df[:train_size]
  test_set = df[train_size:]
  num_itr=int(len(df)/2)
  return train_set,test_set,num_itr

#preprocessing for regression dataset
def preprocessing_r():
  #read the data files into dataframe
  list1=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",
         "quality"]

  #read data into a dataframe
  data1=pd.read_csv("/content/drive/MyDrive/winequality-red.csv")
  data2=pd.read_csv("/content/drive/MyDrive/winequality-white.csv")

  data1.columns=["input"]
  data1=data1["input"].str.split(";",expand=True)
  data1.columns=list1

  data2.columns=["input"]
  data2=data2["input"].str.split(";",expand=True)
  data2.columns=list1

  #cobime all dataframe itno one dataframe 
  final=[data1,data2]
  df=pd.concat(final)
  #convert column to type int  
  df["quality"]=df["quality"].astype(int)

  #scale the numeric values using min max scaler to normalize and get the target output
  scale=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

  #convert values to float
  df[["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]=df[["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].astype(float)

  #perform min max scaling
  t=1
  for i in scale:
    df[i]=(df[i]-df[i].mean())/(df[i].max()-df[i].mean())
    t*=df[i]

  #shuffle dataframe and split dataframe in 70% train and 30% test
  df= df.sample(frac=1).reset_index(drop=True) 
  train_size = int(0.7 * len(df)) 
  train_set = df[:train_size]
  test_set = df[train_size:]
  num_itr=int(len(df)/2)
  
  return train_set,test_set,num_itr

# hyperbolic activation function
def h_t(x):
  return (1-np.exp(np.dot(-2,x)))/(1+np.exp(np.dot(-2,x)))

#derivative of  hyperbolic activation function 
def d_h_t(x):
  return (4*np.exp(np.dot(2,x)))/((1 + np.exp(np.dot(2,x)))**2)

#function to plot mse, rmse and accuracy graphs
def plot_graphs(epochs,value,data,color,dataname):
  fig=plt.figure()
  plt.xlabel("Epochs {}".format(epochs))
  plt.title("Learning curve")
  plt.ylabel("{}".format(value))
  plt.plot(data,color="{}".format(color))
  plt.show()
  fig.savefig("{}_{}".format(dataname,value))

#function to find the testing accuracy for classification and regression dataset
#if value equal to 1 then it will give classification results else regression results
def test_nn_bp_c_r(test,w1,w2,w3,value):
  #counter to calculate accuracy and lists to collect error after each iteration
  correct=0
  error=[]
  mse=[]
  rmse=[]
  for i in range(0,len(test)):
    #fetch data from testing set leaving last column which is target output
    x=np.append(test[i][0:-1],1)
    #get the target ouput which is the last column
    d=test[i][-1]
    #three hidden layers
    hd1=np.append(h_t(np.dot(w1,x)),1)
    hd2=np.append(h_t(np.dot(w2,hd1)),1)
    o=h_t(np.dot(w3,hd2))
    #append error in the list
    error.append(np.square(d-o))
    if o>0:
      o=1
    else: 
      o=-1
    if d==o:
      correct+=1
  #calculate the performance measures
  acc=(correct/len(test))*100
  mse=np.mean(error)
  rmse=np.sqrt(mse)
  #if value is 1 then return performance measures for classification else for regression
  if value==1:
    print("classfication mse {} accuracy {} for testing".format(mse,acc))
  else:
    print("regression mse {} rmse {} for testing".format(mse,rmse))


#train the neural network with backpropogation for classification and rgression dataset
def train_nn_bp_c_r(train,w1,w2,w3,dw1,dw2,dw3,alpha,mse_limit,num_epoch,n,n_hd,value):
  #lists for to collect error and accuracy ouput after every epoch
  mse_error=[]
  rmse_error=[]
  accuracy=[]
  epochs=1
  #iterate for the number of epochs
  for epoch in range(0,num_epoch):
    print("epoch {}".format(epoch+1))
    correct=0
    error=[]
    #shuffle the dataset
    train=np.random.permutation(train)
    for i in range(0,len(train)):
      #forward propagation
      #fetch from training data exclude last element as its the target output
      x=np.append(train[i][0:-1],1) 
      #get the target output which is the last element or column of training data
      d=train[i][-1]
      #three hidden layers
      hd1=np.append(h_t(np.dot(w1,x)),1)
      hd2=np.append(h_t(np.dot(w2,hd1)),1)
      o=h_t(np.dot(w3,hd2))
      #calculate the error
      e=d-o
      #append the error  in list
      error.append(e)
      #reshape the vlaues
      o=o.reshape(-1,1)
      hd2=hd2.reshape(-1,1)
      hd1=hd1.reshape(-1,1)
      #back propagation
      d_ou = e*d_h_t(np.dot(w3,hd2))
      d_hd2 = d_h_t(np.dot(w2,hd1))*np.dot(w3[:,0:n_hd].T,d_ou)
      d_hd1 = d_h_t(np.dot(w1,x)).reshape(-1,1)*np.dot(w2[:,0:n_hd].T,d_hd2[:,0:n_hd-1])
      d_w1 = n*d_hd1*x.T
      d_w2 = n*d_hd2*hd1.T
      d_w3 = n*d_ou*hd2.T
      #weight updates
      bw1 = w1 + alpha*dw1 + d_w1
      bw2 = w2 + alpha*dw2 + d_w2
      bw3 = w3 + alpha*dw3 + d_w3
      dw1=d_w1
      dw2=d_w2
      dw3=d_w3
      w1=bw1
      w2=bw2
      w3=bw3 
      if o>0:
        op=1
      else: 
        op=-1
      if d==op:
        correct+=1
    #decrease learning rate by some value in every eopchs 
    n = n*(1/(1+(n/epochs*epochs)))
    mse=np.mean(np.square(error))
    rmse=np.sqrt(mse)
    acc=(correct/len(train))*100
    mse_error.append(mse) 
    rmse_error.append(rmse)
    accuracy.append(acc)
    epochs+=1 
    #if value is equal to 1 retturn performance results for classification else for regression
    if value==1:
      print("classification mse {} accuracy {} after epoch {} for training".format(mse,acc,epoch+1))
    else:
      print("regression mse {} rmse {} after epoch {} for training".format(mse,rmse,epoch+1)) 
    #termination conditions   
    if mse<mse_limit or epoch>num_epoch:
      print("if conditions satisfed plot and return weights")
      break
  #if value is equal to 1 plot graphs for classification else for regression
  if value==1:
    plot_graphs(num_epoch,"mse error",mse_error,"midnightblue","mse")
    plot_graphs(num_epoch,"accuracy",accuracy,"aquamarine","accuracy")
  else:
    plot_graphs(num_epoch,"mse error",mse_error,"midnightblue","mse")
    plot_graphs(num_epoch,"rmse error",rmse_error,"crimson","rmse")
  return w1,w2,w3

#function to excute above function in a sequence with the hyperparameters 
def main():
  #get train and test sets 70% training and 30% testing for classification
  train_set,test_set,num_itr=preprocessing_c()
  #initialize parameters for classifcaion
  n_in=4     #number of input neuron
  n_hd=6    #number of hidden neurons
  n_ou=1     #number of output neuron
  #intialize weights for three hidden layers
  w1=np.random.random_sample(size=[n_hd,n_in+1])
  dw1=np.zeros((n_hd,n_in+1))
  w2=np.random.random_sample(size=[n_hd,n_hd+1])
  dw2=np.zeros((n_hd,n_hd+1))
  w3=np.random.random_sample(size=[n_ou,n_hd+1])
  dw3=np.zeros((n_ou,n_hd+1))
  #number of epoch
  num_epoch=10
  #mse threshold
  mse_limit=1E-3
  #alpha
  alpha=0.8
  #learning rate 
  n=0.001
  #training 
  #shuffle data
  train_set=train_set.sample(frac=1).reset_index(drop=True) 
  #convert to numpy array
  train=train_set[['Temperature', 'Humidity', 'Light', 'CO2','target']].to_numpy()
  print("length of train {}".format(len(train)))
  #start time 
  st=time.time()
  cw1,cw2,cw3=train_nn_bp_c_r(train,w1,w2,w3,dw1,dw2,dw3,alpha,mse_limit,num_epoch,n,n_hd,1)
  print("time required for training on classification dataset {:.4f}".format(time.time()-st))
  #testing
  #convert to numpy
  test=test_set[['Temperature', 'Humidity', 'Light', 'CO2','target']].to_numpy()
  print("length of test {}".format(len(test)))
  #start time
  st=time.time()
  test_nn_bp_c_r(test,cw1,cw2,cw3,1)
  print("time required for testing on classification datset {:.4f}".format(time.time()-st))

  #initialize parameters for regrssion
  n_in=11     #number of input neuron
  n_hd=6    #number of hidden neurons
  n_ou=1     #number of output neuron
  #initialize weights for three hidden layers
  w1=np.random.random_sample(size=[n_hd,n_in+1])
  dw1=np.zeros((n_hd,n_in+1))
  w2=np.random.random_sample(size=[n_hd,n_hd+1])
  dw2=np.zeros((n_hd,n_hd+1)) 
  w3=np.random.random_sample(size=[n_ou,n_hd+1])
  dw3=np.zeros((n_ou,n_hd+1)) 
  #number of epoch
  num_epoch=10
  #mse threshold
  mse_limit=1E-3
  #alpha value
  alpha=0
  #learning rate 
  n=0.001
  #get the regression data 70% training and 30% testing
  train_set,test_set,num_itr=preprocessing_r()
  #training
  #convert to numpy array
  train=train_set[["fixed_acidity","volatile_acidity","citric_acid",
                 "residual_sugar","chlorides","free sulfur dioxide",
                 "total sulfur dioxide","density","pH",
                 "sulphates","alcohol","quality"]].to_numpy()
  print("length of train {}".format(len(train)))
  #start time
  st=time.time()
  rw1,rw2,rw3=train_nn_bp_c_r(train,w1,w2,w3,dw1,dw2,dw3,alpha,mse_limit,num_epoch,n,n_hd,0)
  print("time required for training on regression dataset {:.4f}".format(time.time()-st))
  #testing
  #convert to numpy array
  test=test_set[["fixed_acidity","volatile_acidity","citric_acid",
                 "residual_sugar","chlorides","free sulfur dioxide",
                 "total sulfur dioxide","density","pH",
                 "sulphates","alcohol","quality"]].to_numpy()
  print("length of test {}".format(len(test)))
  #start time 
  st=time.time()
  test_nn_bp_c_r(test,rw1,rw2,rw3,0)
  print("time required for testing on regression datset {:.4f}".format(time.time()-st))

main()