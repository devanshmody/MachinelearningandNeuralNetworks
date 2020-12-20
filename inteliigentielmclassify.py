import numpy as np
import pandas as pd
import time,os
import matplotlib.pyplot as plt
from numpy_pso import *

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
  correct=np.count_nonzero(np.where(t==y,1,0))#this is used to find accuracy of elements correctly calssified
  error=1-correct #calculate error
  return error

#activation function 
def sigmoid(x,w,bias):
  z = np.dot(x,w)+bias
  return 1 / (1 + np.exp(-z))

#plot the graphs for acuuracy and number of neurons
def graph(n,tr_acc,tt_acc):
  plt.figure()
  plt.xlabel("Number of hidden neurons {}".format(n))
  plt.title("Learning curve")
  plt.ylabel("Accuracy")
  plt.plot(tr_acc,color="DARKBLUE",label="Training")
  plt.plot(tt_acc,tt_acc,'ro',label="Testing")
  plt.legend()
  plt.show()

#function to preprocess data according to format required for traning
def preprocessing():
  #read the data files into dataframe
  data1=pd.read_csv("datatraining.txt")
  data2=pd.read_csv("datatest.txt")
  data3=pd.read_csv("datatest2.txt")

  #cobime all dataframe itno one dataframe of 20560 rows
  final=[data1,data2,data3]
  df=pd.concat(final)

  #scale the numeric values using min max scaler to normalize and get the target output
  t=1
  list1=["Temperature","Humidity","Light","CO2"]
  for i in list1:
    df[i]=(df[i]-df[i].mean())/(df[i].max()-df[i].mean())
    t*=df[i]

  df["target"]=df["Occupancy"]
  df=df[["Temperature","Humidity","Light","CO2","target"]].round(2)

  #shuffle dataframe and split dataframe in 70% train and 30% test
  df= df.sample(frac=1).reset_index(drop=True) 
  train_size = int(0.7 * len(df)) 
  train_set = df[:train_size]
  test_set = df[train_size:]
  
  return train_set,test_set

#Ielm with pso for classification
def IELM(train_data,hidden_size,best_accuracy):
  #convert target and input features to numpy array
  t=train_data["target"].to_numpy()
  df=train_data[["Temperature","Humidity","Light","CO2"]].to_numpy()
  ip_beta=[]
  ip_bias=[]
  ip_weights=[]
  acc=[]
  #here hidden_size is total number of times the loop should iterate and at each iteration a new neuron is added
  for i in range(0,hidden_size):
    #only for our first neuron input weights,bias is generated randomly then from 2 neuron onwards pso is used
    if i==0:
      bias = np.random.normal(size=[1])
      input_weights=np.random.normal(size=[train_data.shape[1]-1,1])
    else:
      #from 2 neuron onwards till then number of neurons get weights,bias from the output of pso 
      #the parameters passed to Swarm are
      #first is size of particle,dimension,Min and Max value(range) of dimension,Min and Max value(range) of velocity,Min and Max value(range) of interia weight,c[0] is cognitive parameter, c[1] is social parameter and last is fitness value
      s = Swarm(10,5,(-5,10), (-2,2), (-1,1), (0.5,0.3))
      #we call optimize function with parameters input,target output,pause between two prints and iterations
      s.optimize(ielm,df,t,1,2000)
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
    xt=np.transpose(x)
    beta = np.dot(np.linalg.inv(np.dot(xt,x)),np.dot(xt,t))#clculate beta value
    ip_beta.extend(beta)
    y=np.dot(x,beta)#multipy beta with output of sigmoid to get the output value
    #here for only finding accuracy of classifiction i take new variale pred which contains values 1 and 0
    pred=np.where(y>0.5,1,0)
    #below method is used to find elemnts which are classified correctly
    #if two values matches it returns one else and at the end i count total number ones
    correct=np.count_nonzero(np.where(t==pred,1,0))
    accuracy=(correct/len(t))*100
    acc.append(accuracy)
    #if accuracy is greater than best accuracy get values of bias,weights and beta till the best accuracy achieved
    if accuracy > best_accuracy:
      best_accuracy=accuracy
      final_weights=ip_weights
      final_bias=ip_bias
      final_beta=ip_beta
      print("best Accuracy of classification {}".format(best_accuracy))
      print("best weights and bias for number of hidden neurons {}".format(i+1))
    #if accuracy becomes 100 this is the max accuracy so return bias,weights,beta values till best accuracy
    if accuracy == 100.0:
      return best_accuracy,final_weights,final_bias,final_beta,acc
    # residual error is calculted and target output is updated for next neuron
    t=t-y
    #its classification and i have class information as 1 and 0 so i convert it back to 1 and 0
    t=np.where(t>0.5,1,0)
  return best_accuracy,final_weights,final_bias,final_beta,acc

#this fucntion is used for testing purpose
def predict(test_data,final_weights,final_bias,final_beta):
  t1=test_data["target"].to_numpy()
  df1=test_data[["Temperature","Humidity","Light","CO2"]].to_numpy()
  #print(len(final_beta),len(final_bias),len(final_weights))
  #convert values to numpy array and reshape only weights
  #here final_beta contains all the values but we want values till which it gave best results
  #so len(final_bias) contains total number of values which gave best results so we select the values till the len(final_bias) 
  final_beta=final_beta[0:len(final_bias)]
  final_beta=np.array(final_beta)
  final_bias=np.array(final_bias)
  final_weights=np.array(final_weights).reshape(4,-1)
  #print(len(final_beta),len(final_bias),len(final_weights))
  out = sigmoid(df1,final_weights,final_bias)
  out = np.dot(out,final_beta)
  out=np.where(out>0.5,1,0)
  #method to find the number of classifitions which are correct 
  correct=np.count_nonzero(np.where(t1==out,1,0))
  accuracy=(correct/len(t1))*100
  
  return accuracy

#all the above given steps are executed in a sequence and put together in main 
def main():
  train_data,test_data=preprocessing()
  #shuffle data
  train_data=train_data.sample(frac=1).reset_index(drop=True) 
  #size of hidden neurons is passed as for loop parameter in IELM_PSO function 
  #so for loop in function iterates for 1k times and at each iteration neuron are addded iteratively till 1k or error beecomes zero
  hidden_size = 1000 #size of hidden neurons
  best_acc=0 #intial accuracy
  print("length of train_data {} and test_data {}".format(len(train_data),len(test_data)))

  st=time.time()#star time
  #call the function for train set
  best_accuracy,final_weights,final_bias,final_beta,acc=IELM(train_data,hidden_size,best_acc)
  print("=====Accuracy and time for Training=====")
  print("Accuracy For Training  %.4f"%best_accuracy)
  print("Time Elapsed For Training",time.time()-st)

  st=time.time()#start time
  #calculate the function for test set
  t_acc=predict(test_data,final_weights,final_bias,final_beta)
  print("=====Accuracy and time for Testing=====")
  print("Accuracy For Testing %.4f"%t_acc)
  print("Time Elapsed For Testing",time.time()-st)
  
  #plot the graph
  graph(hidden_size,acc,t_acc)

main()

