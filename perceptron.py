import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

'''function to generate half moon data where
r is radius, w is width , d is distance,t is theta 
n is number of data to be generated '''

def generate_hm_data(r, w, d,n):
    #check if number is even or not if not then make it even 
    if n%2!= 0:n+= 1  

    s = np.random.random_sample((2,n//2)) #generate samples ie s.shape(2,1500) 

    rad = (r-w//2) + w*s[0,:] 
    t = np.pi*s[1,:]        
      
    x     = rad*np.cos(t)  
    y     = rad*np.sin(t)  
    label = np.ones((1,len(x))) # label for class 1  
      
    x1    = rad*np.cos(-t) + r  
    y1    = rad*np.sin(-t) - d  
    label1= -1*np.ones((1,len(x1))) # label for class -1 
     
    test=[np.append(x,x1),np.append(y,y1),np.append(label,label1)]
    return test

#function to pass parameters to get half moon data and plot it
def get_data(radius,width,distance,traning,test):
  total=traning+test
  
  test = generate_hm_data(radius,width,distance,total)
  dk=pd.DataFrame(test) #convert output into a dataframe 
  dk=dk.T
  dk.columns=["input1","input2","label"]
  dk.reset_index(drop=True,inplace=True)
  pos=dk[dk.label==1.0] #classify 1
  neg=dk[dk.label==-1.0] #classify -1

  plt.figure()
  plt.title('Half Moon')
  plt.xlabel('data points in red belongs to class -1')
  plt.ylabel('data points in green belongs to class 1')
  plt.scatter(pos["input1"],pos["input2"],color="Green",s=5)
  plt.scatter(neg["input1"],neg["input2"],color="Red",s=5)
  plt.show()

  return dk

#perceptron network 
def per_net(value,df,bias,learning_rate,w):
  y=0 #output
  for i in range(0,len(w)):
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
  

  #plt.plot(x)
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
  pos=df[df.label==1]
  neg=df[df.label==-1]

  #lets find equation of line to plot 
  x = -bias / w[0]
  y= -bias/w[1]

  plt.figure()
  plt.title('Half Moon')
  plt.title('Result')
  plt.xlabel('data points in red belongs to class -1')
  plt.ylabel('data points in green belongs to class 1')
  plt.scatter(pos["input1"],pos["input2"],color="Green",s=5)
  plt.plot([x,0],[0,y])
  plt.scatter(neg["input1"],neg["input2"],color="Red",s=5)
  plt.show()

def main():
  st=time.time() #start time
  #set parameters for half moon
  radius,width,distance,traning,test=10,6,1,1000,2000 

  #to get the data generated form the half moon
  test=get_data(radius,width,distance,traning,test) 

  #set the hyperparameters for perceptron 
  epochs,bias,learning_rate=50,0,0.5

  #dpending on input columns required for training  create weight vector 
  w=[0,0]  
  sample=2000 

  #shuffle dataframe
  test = test.sample(frac=1).reset_index(drop=True) 
  df=test[:sample] #use selected data points based on sample for training perceptron
  df["target"]=df["input1"]*df["input2"]
  df[["input1","input2","target"]]=df[["input1","input2","target"]].round(1)
  df["target"]=np.sign(df["target"]) #ifnegative -1 if positve 1 else 0

  #to collect accuracy and error 
  a=[] 
  e=[]
  m=[]
  r=[]

  #train model for number of epochs
  for i in range(0,epochs):
    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe for every epoch
    accuracy,error_rate,mse,rmse,w,bias=per_net(i+1,df,bias,learning_rate,w)
    a.append(accuracy)
    e.append(error_rate)
    m.append(mse)
    r.append(rmse)

  plot_graphs(epochs,"Accuracy",a,"blue")
  plot_graphs(epochs,"Error Rate",e,"Red")
  plot_graphs(epochs,"MSE",m,"Yellow")
  plot_graphs(epochs,"RMSE",r,"Orange")
  plot_line(df,w,bias) 
  print("total time required",time.time()-st)

main()

