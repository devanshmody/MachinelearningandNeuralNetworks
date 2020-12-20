import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

#function to preprocess data according to format required for traning
def preprocessing():
  #read the data files into dataframe
  data1=pd.read_csv("datatraining.txt")
  data2=pd.read_csv("datatest.txt")
  data3=pd.read_csv("datatest2.txt")

  #cobime all dataframe itno one dataframe of 20560 rows
  final=[data1,data2,data3]
  df=pd.concat(final)

  #scale the numeric values using min max scaler and get the target output
  t=1
  list1=["Temperature","Humidity","Light","CO2"]
  for i in list1:
    df[i]=(df[i]-df[i].mean())/(df[i].max()-df[i].mean())
    t*=df[i]

  df["target"]=t
  df["target"]=np.sign(df["target"]) #ifnegative -1 if positve 1 else 0

  #shuffle dataframe and split dataframe in 70% train and 30% test
  df= df.sample(frac=1).reset_index(drop=True) 
  train_size = int(0.7 * len(df)) 
  train_set = df[:train_size]
  test_set = df[train_size:]

  return train_set,test_set

#perceptron network 
def per_net(value,df,bias,learning_rate,w):
  y=0 #output
  for i in range(0,len(w)-4):
    y+=np.dot(df["input{}".format(i+1)],w[i])#to get dot product between input and weigths and sum them
  
  #get output y value 0,-1 or 1
  y+=bias   #add bias to sum
  y=np.sign(y) #if negative assign -1 if zero then zero if positive then 1 its activation function
  df["predicted"]=y 
  df["difference"]=df["target"]-df["predicted"] #calulate diffrence between target and predicted output

  #for calculating acuuracy and error and weight and bias update 
  correct=0
  for j in range(0,len(df)):
    if df["target"][j]!=df["predicted"][j]:
      bias = bias + learning_rate * (df["target"][j]-df["predicted"][j]) 
      for p in range(0,len(w)):
       # print("updating old weights ",w)
        w[p]=w[p]+learning_rate*(df["target"][j]-df["predicted"][j])*(df["input{}".format(p+1)][j])
     # print("updated ne  weights ",w)
    if df["target"][j] == df["predicted"][j]:
      correct+=1
  
  accuracy=(correct / float(len(df)))*100.0
  error_rate=1-(correct / float(len(df)))

  print("======Epoch {}======".format(value))
  print("Accuracy in percent %.4f" % accuracy) #total number of correctly classified/total
  print("Error Rate %.4f" % error_rate) #error rate is number of missclassified/total which is 1-accuracy
  mse=(np.square(df["target"] - df["predicted"])).mean(axis=0)   
  print("MSE error %.4f" % np.sum(mse))
  print("RMSE error %.4f" % np.sqrt(mse))

  return accuracy,error_rate,mse,np.sqrt(mse),w,bias

#function to plot accuracy and error graph
def plot_graphs(epochs,value,data,color):
  plt.figure()
  plt.xlabel("Epochs {}".format(epochs))
  plt.title("Learning curve")
  plt.ylabel("{}".format(value))
  plt.plot(data,color="{}".format(color))
  plt.show()

#function to plot line seprating data that is decision boundary
def plot_line(df,w,bias):
  x=[]
  y=[]
  for i in range(0,len(w)-1):
    x.append(-bias/w[i])
    y.append(-bias/w[i+1])

  pos=df[df.target==1]
  neg=df[df.target<1]
  plt.figure()
  plt.title('Result')
  plt.xlabel('data points in red belongs to class 0')
  plt.ylabel('data points in green belongs to class 1')
  plt.scatter(pos["input1"],pos["input2"],color="Green",s=5)
  plt.scatter(pos["input3"],pos["input4"],color="Green",s=5)
  plt.plot(x,y)
  plt.scatter(neg["input1"],neg["input2"],color="Red",s=5)
  plt.scatter(neg["input3"],neg["input4"],color="Red",s=5)
  plt.show()

#function to run for training and test set
def training_testing(df,epochs,bias,learning_rate):
  #dpending on input columns required for training  create weight vector 
  w=np.zeros(len(df.columns)-4) 

  #to collect accuracy and error 
  a=[] 
  e=[]
  m=[]
  r=[]

  list1=["Temperature","Humidity","Light","CO2"]
  #renaming of columns used to train perceptron 
  for i in range(0,len(list1)):
    df.rename(columns={list1[i]:"input{}".format(i+1)},inplace=True)
    #train model for number of epochs
  for j in range(0,epochs):
    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe for every epoch
    accuracy,error_rate,mse,rmse,w,bias=per_net(j+1,df,bias,learning_rate,w)
    a.append(accuracy)
    e.append(error_rate)
    m.append(mse)
    r.append(rmse)

  #to plot the final output graphs also there can be put inside for loop to measeure output at each epoch 
  plot_graphs(epochs,"Accuracy",a,"blue")
  plot_graphs(epochs,"Error Rate",e,"Red")
  plot_graphs(epochs,"MSE",m,"Yellow")
  plot_graphs(epochs,"RMSE",r,"Orange")
  plot_line(df,w,bias) 

  return accuracy,error_rate,mse,rmse

def main():
  #set the hyperparameters for perceptron 
  epochs,bias,learning_rate=100,0,0.5

  #get train and test sets 70% training and 30% testing
  train_set,test_set=preprocessing()
  st=time.time()
  train_accuracy,train_error_rate,train_mse,train_rmse=training_testing(train_set,epochs,bias,learning_rate)
  print("time required for on traning set",time.time()-st)
  print("train_set accuracy,error_rate,mse,rmse",train_accuracy,train_error_rate,train_mse,train_rmse)
  st=time.time()
  test_accuracy,test_error_rate,test_mse,test_rmse=training_testing(test_set,epochs,bias,learning_rate)
  print("train_set accuracy,error_rate,mse,rmse",test_accuracy,test_error_rate,test_mse,test_rmse)
  print("time required for on testing set",time.time()-st)

main()

