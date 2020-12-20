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

def LMS(data,epochs,w,bias,learning_rate,w_limit,rmse_limit,num_itr):
  counter=20 #set counter to 20 and decrease learning rate gradually
  #create empty list for accuracy,erro rate,mse,rmse,weights
  a=[]
  e=[]
  ms=[]
  rm=[]
  wt=[]
 # print("epochs",epochs)
  for i in range(0,epochs):
    er=[] #to collect all squared errors
    correct=0 #to get correctly classfied data
    yout=0
    op=[]
    y=0

    #shuffle data
    data=data.sample(frac=1).reset_index(drop=True) 
  #  print("data {}".format(i),data)
    t=data["target"].to_numpy()
    df=data[["Temperature","Humidity","Light","CO2"]].to_numpy()
    if i==counter:
      learning_rate=learning_rate/10
      counter=2*counter
   #   print("learning_rate",learning_rate)

    for idx,value in enumerate(df):
      #print(idx,value)
      y=np.dot(w.transpose(),value)+bias
      op.append(np.sign(y))
      yout=np.sign(y)
      w_delta=learning_rate*(t[idx]-y)*value
      wt.append(w)
      w=w+w_delta
      bias=bias+learning_rate*(t[idx]-y)
      er.append(np.square(t[idx]-y))
      if t[idx]==yout:
        correct+=1
      #breaking conditions
      if (np.linalg.norm(w_delta)<w_limit) and  (i >=num_itr):
        print("limit reached1")
        break
    
    accuracy=(correct/float(len(er)))*100.0
    a.append(accuracy)
    error_rate=1-(correct/float(len(er)))
    e.append(error_rate)
    print("======Epoch {}======".format(i+1))
    print("Accuracy in percent %.4f" % accuracy) #total number of correctly classified/total
#   print("Miss Classification Error Rate %.4f" % error_rate) #error rate is number of missclassified/total which is 1-accuracy   
    mse=sum(er)/len(er)
    ms.append(mse)
    print("MSE error %.4f" % mse)
    rmse=np.sqrt(mse)
    rm.append(rmse)
    print("RMSE error %.4f" % rmse)

    if rmse<rmse_limit:
      print("limit reached2")
      break  
    
  return bias,wt,rm,ms,a,e

#function to plot accuracy and error graph
def plot_graphs(epochs,value,data,color,dataname):
  fig=plt.figure()
  plt.xlabel("Epochs {}".format(epochs))
  plt.title("Learning curve")
  plt.ylabel("{}".format(value))
  plt.plot(data,color="{}".format(color))
  plt.show()
  fig.savefig("{}_{}".format(dataname,value))
#function to plot line seprating data that is decision boundary
def plot_line(df,w,bias,dataname):
  pos=df[df.target==1]
  neg=df[df.target<1]
  fig=plt.figure()
  plt.title('Result')
  plt.xlabel('data points in red belongs to class room occupied')
  plt.ylabel('data points in green belongs to class room not occupied')
  plt.scatter(pos["Temperature"][:200],pos["Humidity"][:200],color="Green",s=5)
  plt.scatter(pos["Light"][:200],pos["CO2"][:200],color="Green",s=5)
  plt.scatter(neg["Temperature"][:200],neg["Humidity"][:200],color="Red",s=5)
  plt.scatter(neg["Light"][:200],neg["CO2"][:200],color="Red",s=5)
  plt.plot(w[-1])
  plt.show()
  fig.savefig("{}".format(dataname))

def main():
  #set the hyperparameters for perceptron 
  epochs,learning_rate,w_limit,rmse_limit=1000,0.1,1E-8,1E-6
  bias=np.round(np.random.random_sample(1),2)
  #get train and test sets 70% training and 30% testing
  train_set,test_set,num_itr=preprocessing()
  #print(" initial bias",bias)
  w=np.linspace(-1,1,len(train_set.columns)-1)
  #print("initial weight",w)
  st=time.time() #start time
  tr_bias,tr_w,tr_rmse,tr_mse,tr_accuracy,tr_error_rate=LMS(train_set,epochs,w,bias,learning_rate,w_limit,rmse_limit,num_itr)
  print("time required for on traning set {:.4f}".format(time.time()-st)) #calculating time that is end time subtracted by start time
  print("Accuracy in percent %.4f" % tr_accuracy[-1])
 # print("Miss Classification error rate %.4f" % tr_error_rate[-1])
  print("MSE error %.4f" % tr_mse[-1])
  print("RMSE error %.4f" % tr_rmse[-1])
  #plot the final output graphs 
  plot_graphs(epochs,"Accuracy",tr_accuracy,"darkgreen","training_set")
  #plot_graphs(epochs,"Miss Classification Error Rate",tr_error_rate,"darkorange","training_set")
  plot_graphs(epochs,"MSE",tr_mse,"darkblue","training_set")
  plot_graphs(epochs,"RMSE",tr_rmse,"darkviolet","training_set")
  plot_line(train_set,tr_w,tr_bias,"training_set") 

  st=time.time()#used updated weighs
  tt_bias,tt_w,tt_rmse,tt_mse,tt_accuracy,tt_error_rate=LMS(test_set,epochs,tr_w,tr_bias,learning_rate,w_limit,rmse_limit,num_itr)
  print("time required for on testing set {:.4f}".format(time.time()-st))
  print("Accuracy in percent %.4f" % tt_accuracy[-1])
  #print("Miss Classification error rate %.4f" % tt_error_rate[-1])
  print("MSE error %.4f" % tt_mse[-1])
  print("RMSE error %.4f" % tt_rmse[-1])
  #plot the final output graphs 
  plot_graphs(epochs,"Accuracy",tt_accuracy,"darkgreen","testing_set")
 # plot_graphs(epochs,"Miss Classification Error Rate",tt_error_rate,"darkorange","testing_set")
  plot_graphs(epochs,"MSE",tt_mse,"darkblue","testing_set")
  plot_graphs(epochs,"RMSE",tt_rmse,"darkviolet","testing_set")
  plot_line(test_set,tt_w,tt_bias,"testing_set")

main()

