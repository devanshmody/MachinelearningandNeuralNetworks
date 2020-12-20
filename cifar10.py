import numpy as np
import pandas as pd
import time,re,zipfile,os,gzip,pickle,tarfile
import matplotlib.pyplot as plt

#extract tar file from gz file then extract contents of tar file
gunzip('/content/drive/My Drive/cifar-10-python.tar.gz')
my_tar = tarfile.open("/content/drive/My Drive/cifar-10-python.tar", 'r')
my_tar.extractall("/devvoice") # specify which folder to extract to
my_tar.close()

#the file which contains data is in pickle format so unpickle the file
def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

#plot the accuracy graph
def graph(n,tr_acc,tt_acc):
  plt.figure()
  plt.xlabel("Number of hidden neurons {}".format(n))
  plt.title("Learning curve")
  plt.ylabel("Accuracy")
  plt.plot(tr_acc,color="DARKBLUE",label="Training")
  plt.plot(tt_acc,tt_acc,'ro',label="Testing")
  plt.legend()
  plt.show()

#preprocess the input data in required format
def preprocessing(filename):
  #the name of the file is given as input and its converted to required format
  data=unpickle("/devvoice/cifar-10-batches-py/{}".format(filename))
  #the unpickle output returns a dictionary so extract key and values from a dicitonary
  key_list = list(data.keys()) 
  val_list = list(data.values())  
  #create a empty dataframe and add values to it
  df=pd.DataFrame()
  df["labels"]=val_list[1]
  df["data_val"]=list(val_list[2])
  df["filenames"]=val_list[3]
  #normalize the input array for each row of column data_val
  for i in range(0,len(df)):
    norm = np.linalg.norm(df["data_val"][i])
    normal_array = df["data_val"][i]/norm
    df["data_val"][i]=normal_array

  #print("length of {} is {}".format(filename,len(df)))
  #convert labels to type int
  df=df[["labels","data_val"]]	
  df["labels"]=df["labels"].astype(int)

  #one hot encoding on the labels 
  #so we get output like this 
  #for calss0 we have one hot encode output as [input array,1,zeros for remaining clases]
  #for class1 we have one hot encode output as [input array,0,1,zeroos for remaining]
  #similarly we encode for o to 10 classes 
  list2=df["labels"].unique()
  for i in list2:
    #print(i)
    df["class{}".format(i)]=np.where(df["labels"].values == i,1,0)

  #shuffle the dataset
  df=df.sample(frac=1).reset_index(drop=True) 
  #drop the columns labels as we have our one hot encoded columns 
  df.drop(columns=["labels"],axis=1,inplace=True)
  df.reset_index(drop=True,inplace=True)
  #reset the indes
  return df

#sigmoid activation function
def sigmoid(x,w,bias):
  z = np.dot(x,w)+bias
  return 1 / (1 + np.exp(-z))

#ielm for cifar10
def IELM_CIFR(train_data,hidden_size,best_accuracy):
  #df contains a list of array 
  df=train_data["data_val"].to_list()
  #t is the target class which is converted to numpy array
  #it contains one hot encode values like this
  #[1,remaining 10 zeros] for class0 
  #[0,1,remaining zeros] for class1
  #[remaning zeros, last value 1]which indiacates class 9
  t=train_data[['class0','class1','class2','class3','class4','class5','class6','class7','class8','class9']].to_numpy()
  acc=[] #append accuracy in this 
  #iterate for the number of hidden_size and in each iteration bias,weights and neuron is added iteratively
  #for first iteration one neuron and weights and bias taken ramdonly for that neuron then after each iteration keep on adding
  for i in range(0,hidden_size):
    #get bias randomly
    bias = np.random.normal(size=[i+1])
    #get weights randomly and add each neuron iteratively 
    input_weights = np.random.normal(size=[3072,i+1])
    x=sigmoid(df,input_weights,bias)#get sigmoid output
    xt=np.transpose(x)#get transpose of output
    beta = np.dot(np.linalg.inv(np.dot(xt,x)),np.dot(xt,t))#calculate beta
    y=np.dot(x,beta)#mutiply beta with sigmoid output and get predicted output which contains calsses form 0 to 10 for input rows
    pred=np.where(y>0.5,1,0)#here for comparison betwee target and predicted i convert to 0 and 1 
    #for example if model has predicted class0 for input then arrya will look like this [1,0,remaining zeros,0] for 10 classes
    #then the array will be compared with the target array
    #i count number of samples correctly classified for finding accuracy
    #instead of taking a for loop and then comapring iteratively for target==prdicted and then incrementing counter 
    #if target equal to predicted which is a slow process 
    #i use list comprehension and i take arrays and get dot product and then count maximum number of 1 and return it
    #this fucntion works as follows if i have 100 values whih predict class 0 and target values which predict class zero
    #my array for target will look like [1,0,0,0,0,0,0,0,0,0] and for predict will look [1,0,0,0,0,0,0,0,0,0] both will be same
    #so when i take dot product i will get one so i then count number of 1's in dot product recived and return it 
    #so if 100 rows of both target and predict match i will get count 100 that is my 100 rows are corrrectly classified
    #if any rows dosent match with target then dot product will be zero 
    #so its role is to count number of target==predicted that is number of elements which are correctly classified
    #but by using below method of taking dot product and counting ones , instead of using for loop and if target==predicted
    #below way is faster then for loop method and saves time
    counter = [np.dot(x,y) for (x, y) in zip(t,pred)].count(1)
    #print("count={} neurons={}".format(counter,i+1))
    #calculate and append accuracy
    accuracy=(counter/len(t))*100 
    acc.append(accuracy)
    #save best accuracy intial accuracy is zero then based on the best accuray the new values are stored
    if accuracy > best_accuracy:
      best_accuracy=accuracy
      final_weights=input_weights
      final_bias=bias
      final_beta=beta
      #print("best Accuracy of classification {}".format(accuracy))
      #print("best weights and bias for number of hidden neurons {}".format(i+1))
      #print("best weights {}",final_weights)
      #print("best bias {}",bias)

    #if accuracy is equal to 100 as this is the maximum accuracy return final weights,bias,beta and accuracy
    if accuracy == 100.0:
      return best_accuracy,final_weights,final_bias,final_beta,acc
  #update our target output or residual error
  t=t-y
  #as in classification we have classes in our output i conver back the target to 0 and 1 like it was in one hot 
  #and it gives me output in class format that is it gives [1,remainin zeros] for class0 and like wise for class1[0,1,remaining zeros] and so on till 10th class
  t=np.where(t>0.5,1,0)

  return best_accuracy,final_weights,final_bias,final_beta,acc

#this fucntion is used for testing
def predict(test_data,final_weights,final_bias,final_beta):
  #convert data_val to list and input or features to numpy array
  df1=test_data["data_val"].to_list()
  t1=test_data[['class0','class1','class2','class3','class4','class5','class6','class7','class8','class9']].to_numpy() 
  out = sigmoid(df1,final_weights,final_bias)
  out = np.dot(out,final_beta)
  out=np.where(out>0.5,1,0)
  #fucntion to count number of target==predicted that is number of elements which are correctly classified
  counter = [np.dot(x,y) for (x, y) in zip(t1,out)].count(1)
  accuracy=(counter/len(t1))*100
  #print("Number of hidden neurons {}".format(len(final_weights[0])))
  #print("Accuracy %.4f" % accuracy)

  return accuracy

def main():
  #get the data for taining and combine the whole data into one dataframe
  train_path=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
  train=[]
  for i in train_path:
    train.append(preprocessing(i))

  train_data=pd.concat(train)
  train_data.reset_index(drop=True,inplace=True)

  test_data=preprocessing("test_batch")
  test_data.reset_index(drop=True,inplace=True)

  hidden_size = 1000 #size of hidden neurons which are used in loop of ielm cifr function
  #loop runs depending on hidden_size and neurons are added in each iteration
  best_acc=0
  print("length of train {} and test {}".format(len(train_data),len(test_data)))

  st=time.time()#star time
  #for training
  best_accuracy,final_weights,final_bias,final_beta,acc=IELM_CIFR(train_data,hidden_size,best_acc)
  print("=====Accuracy and time for Training=====")
  print("Accuracy For Training  %.4f"%best_accuracy)
  print("Time Elapsed For Training",time.time()-st)

  #print(final_weights,final_bias,final_beta)
  st=time.time()#start time
  #the final weights,bias and beta we get we use it for testing
  t_acc=predict(test_data,final_weights,final_bias,final_beta)
  print("=====Accuracy and time for Testing=====")
  print("Accuracy For Testing %.4f"%t_acc)
  print("Time Elapsed For Testing",time.time()-st)

  #plot the graph
  graph(hidden_size,acc,t_acc)

main()

