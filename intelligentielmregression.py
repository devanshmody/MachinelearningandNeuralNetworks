import numpy as np
import pandas as pd
import time,os
import matplotlib.pyplot as plt
from numpy_pso import *

#activation function 
def sigmoid(x,w,bias):
  z = np.dot(x,w)+bias
  return 1 / (1 + np.exp(-z))
  
#plot the graphs for rmse error and number of neurons
def graph(n,rmse_e,rmse):
  plt.figure()
  plt.xlabel("Number of hidden neurons {}".format(n))
  plt.title("Learning curve")
  plt.ylabel("RMSE Error")
  plt.plot(rmse_e,color="DARKBLUE",label="Training")
  plt.plot(rmse,rmse,'ro',label="Testing")
  plt.legend()
  plt.show()

#we are optimizing ielm so this fucntion returns optimise weights and bias
def ielm(p,df,t):
  #here i take first element of array as bias and remaining as weights and vstack weights 
  bias = np.array(p[0])
  i_w = np.array(p[1:])
  input_weights=np.vstack(i_w)
  x=sigmoid(df,input_weights,bias)#gives output of sigmoid fucntion
  xt=np.transpose(x)#generate transpose
  beta = np.dot(np.linalg.inv(np.dot(xt,x)),np.dot(xt,t))#clculate beta value
  y=np.dot(x,beta) #multipy beta with output of sigmoid to get the output value
  rmse=np.sqrt((np.square(t-y)).mean())#calculates error 
  return rmse

#function to preprocess data according to format required for traning
def preprocessing():
  #read the data files into dataframe
  list1=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",
         "quality"]

  data1=pd.read_csv("winequality-red.csv")
  data2=pd.read_csv("winequality-white.csv")

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
  
  return train_set,test_set

#IELM with PSO for regrssion
def IELM(train_set,hidden_size,error):
  #shuffling once only the dataset before training begins
  train_set=train_set.sample(frac=1).reset_index(drop=True)
  #convert target output and input values of features to numpy array  
  t=train_set["quality"].to_numpy()
  df=train_set[["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].to_numpy()
  ip_size=df.shape[1]#this is the size of the inputs it used for creating weights along with number of neurons
  rmse_e=[]#append rmse error in this list
  ip_beta=[]
  ip_bias=[]
  ip_weights=[]
  #here hidden_size is total number of times the loop should iterate and at each iteration a new neuron is added
  for i in range(0,hidden_size):
    #only for first neuron we generate weights,bias randomly and then onwards we apply pso and generate weights and bias 
    if i==0:
      #assign input bias and weights randomly
      bias = np.random.normal(size=[1]) 
      input_weights = np.random.normal(size=[ip_size,1])
    else:
      #from 2 neuron onwards we apply PSO to generate weights and bias
      #below are the parameters passed to Swarm are  
	#first is size of particle,dimension,Min and Max value(range) of dimension,Min and Max value(range) of velocity,Min and Max value(range) of interia weight,c[0] is cognitive parameter, c[1] is social parameter and last is fitness value
      s = Swarm(10,12,(-5,10), (-2,2), (-1,1), (0.5,0.3))
      #we call optimize function with parameters input,target output,pause between two prints and iterations
      s.optimize(ielm,df,t,100,2000)
      #print("Best Fitness/Min Loss: ", s.gbest)
      #print("Best position/Best weights: ", s.gbestpos)
      #s.gbestpos gets the values in which i take first value as bias and remaining values as weights
      bias=np.array(s.gbestpos[0])
      ip_bias.append(bias)
      con=np.array(s.gbestpos[1:])
      #convert the output to required numpy array format
      input_weights=np.vstack(con)
      ip_weights.append(con)
    x=sigmoid(df,input_weights,bias)#gives output of sigmoid fucntion
    xt=np.transpose(x)#generate transpose
    beta = np.dot(np.linalg.inv(np.dot(xt,x)),np.dot(xt,t))#clculate beta value
    ip_beta.append(beta)
    y=np.dot(x,beta) #multipy beta with output of sigmoid to get the output value
    rmse=np.sqrt((np.square(t-y)).mean()) #calculate the mean square error between target and y output
    rmse_e.append(rmse) #append error
    #here initial error is taken 1 if we get error less than the given error we take the reduced error 
    #so we take values of bias,weights and beta which gives us a reduced error for the number of neurons
    #if we get error value 0.01 for number of neurons 1k then we take values of beta,weights and bias till 1k we continue till we keep on getting reduced error and stop if error is equal to 0.0 as error cant be reduced further
    if rmse < error:
      error=rmse
      final_weights=ip_weights
      final_bias=ip_bias
      final_beta=ip_beta
      print("RMSE error %.4f" % rmse)
      print("best weights and bias for number of hidden neurons {}".format(i+1))
    
    #if error reaches zero we halt as error cant be reduced further and we return best bias,weights ad beta else loop continues
    if error==0.0:
      return rmse_e,final_weights,final_bias,final_beta,error
    
    #here I update target output or residual error for next layer or next neuron
    t=t-y
  
  return rmse_e,final_weights,final_bias,final_beta,error

#this fucntion is used for testing
def predict(test_set,final_weights,final_bias,final_beta):
  #we convert our target and inputs to numpy array
  test_set_y=test_set["quality"].to_numpy()
  test_set_data=test_set[["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].to_numpy()
  #print(len(final_beta),len(final_bias),len(final_weights))
  #convert values to numpy array and reshape only weights
  #here final_beta contains all the values but we want values till which it gave best results
  #so len(final_bias) contains total number of values which gave best results so we select the values till the len(final_bias) 
  final_beta=final_beta[0:len(final_bias)]
  final_beta=np.array(final_beta)
  final_bias=np.array(final_bias)
  final_weights=np.array(final_weights).reshape(11,-1)
  #print(len(final_beta),len(final_bias),len(final_weights))
  #we calculate output using sigmoid activation function
  out = sigmoid(test_set_data,final_weights,final_bias)
  out = np.dot(out,final_beta)
  #we calculate rmse error
  rmse=np.sqrt((np.square(test_set_y-out)).mean())

  return rmse

#all the above given steps are executed in a sequence and put together in main 
def main():
  #get train and test data set 
  train_set,test_set=preprocessing()
  print("length of data train_data {} test_data {}".format(len(train_set),len(test_set)))
  #size of hidden neurons is passed as for loop parameter in IELM function 
  #so for loop in function iterates for 1k times and at each iteration neuron are addded iteratively till 1k or error beecomes zero
  hidden_size = 1000 
  error=1 # initial error
  st=time.time()#star time
  #call the fucntion train set
  rmse_e,final_weights,final_bias,final_beta,error=IELM(train_set,hidden_size,error)
  print("=====RMSE and time for Training=====")
  print("RMSE Error For Training  %.4f"%error)
  print("Time Elapsed For Training",time.time()-st)

  #call the function for test set
  st=time.time()#start time
  rmse=predict(test_set,final_weights,final_bias,final_beta)
  print("=====RMSE and time for Testing=====")
  print("RMSE Error For Testing  %.4f"%rmse)
  print("Time Elapsed For Testing",time.time()-st)

  #plot the graph
  graph(hidden_size,rmse_e,rmse)

main()



