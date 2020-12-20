import numpy as np
import pandas as pd
import time,re,os
import matplotlib.pyplot as plt

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
  num_itr=int(len(df)/2)
  
  return train_set,test_set,num_itr

#sigmoid activation function
def sigmoid(x,w,bias):
  z = np.dot(x,w)+bias
  return 1 / (1 + np.exp(-z))

#incremental elm for regrssion
def IELM(train_set,hidden_size,error):
  #shuffling once only the dataset before training begins
  train_set=train_set.sample(frac=1).reset_index(drop=True)
  #convert target output to numpy array 
  t=train_set["quality"].to_numpy()
  #convert input values or features to numpy array
  df=train_set[["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].to_numpy()
  ip_size=df.shape[1] #this is the size of input columns or features
  rmse_e=[]#append rmse error in this list
  #print("size",ip_size)
  #here hidden_size is total time the loop should iterate and at each iteration a new neuron is added
  for i in range(0,hidden_size):
    #assign input bias and weights randomly
    bias = np.random.normal(size=[i+1])
    #here thre is size=[ip_size,i+1] creates neuron at each iteration that is for i=0 1 neuron and onwards on each iteration
    #for first iteration bias will have one value and as its one neuron only weights will be generated for one neuron
    #then for second iteration i+1 that is 2 neurons are added and so on neuron are added till the size of the loop or if
    #we get 0 rmse error
    input_weights = np.random.normal(size=[ip_size,i+1])
    x=sigmoid(df,input_weights,bias)#gives output of sigmoid fucntion
    xt=np.transpose(x)#generate transpose
    beta = np.dot(np.linalg.inv(np.dot(xt,x)),np.dot(xt,t)) #clculate beta value
    y=np.dot(x,beta) #multipy beta with output of sigmoid to get the output value
    rmse=np.sqrt((np.square(t-y)).mean()) #calculate the mean square error between target and y output
    rmse_e.append(rmse) #append error
    
    #here initial error is taken 1 if we get error less than the given error we take the reduced error 
    #so after adding more nuerons if we take the bias,weights and beta which gives us a reduced error 
    if rmse < error:
      error=rmse
      final_weights=input_weights
      final_bias=bias
      final_beta=beta
      print("RMSE error %.4f" % rmse)
      print("best weights and bias for number of hidden neurons {}".format(i+1))
    
    #if error reaches zero we halt as error cant be reduced further and we return best bias,weights ad beta else loop continues
    if error==0.0:
      return rmse_e,final_weights,final_bias,final_beta,error
    
    #update our target output or residual error
    t=t-y
  
  return rmse_e,final_weights,final_bias,final_beta,error

#this fucntion is used for testinf
def predict(test_set,final_weights,final_bias,final_beta):
  #we convert our target and inputs to numpy array
  test_set_y=test_set["quality"].to_numpy()
  test_set_data=test_set[["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
         "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].to_numpy()
  
  #we calculate output using sigmoid activation function
  out = sigmoid(test_set_data,final_weights,final_bias)
  out = np.dot(out,final_beta)
  #we calculate rmse error
  rmse=np.sqrt((np.square(test_set_y-out)).mean())

  return rmse

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

#all the above given steps are executed in a sequence and put together in main 
def main():
  #get train and test data set 
  train_set,test_set,num_itr=preprocessing()
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

