#extract dataset1.zip file contents
import zipfile
import numpy as  np
import re,time,os,random

zip_ref = zipfile.ZipFile("/content/drive/My Drive/dataset1.zip", 'r')
zip_ref.extractall("/devvoice")
zip_ref.close()

#function is used to convert images to vector
def img_vector(filename):
    return_vect = np.zeros((1,1024))
    fr = open("/devvoice/training_validation/"+filename,"r")
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0,32*i+j] = int(line_str[j])

    return return_vect[0]

#to get the data from the folder for class 0 and 6 and convert image to vector
def preprocess_data(filename):
    data_dir=os.listdir(filename)
#    print("len data_dir",len(data_dir))
    
    trv_data=[]
    for i in data_dir:
        if re.search('class_0',i,re.IGNORECASE):
            trv_data.append((img_vector(i),0))
    
        if re.search('class_6',i,re.IGNORECASE):
            trv_data.append((img_vector(i),6))
          
#        print("len data",len(trv_data)) 
        
    return trv_data
    
#function which counts frequency of class occuring in list and class predicted by classifier
def countfreq(my_list): 
    # count frequncy of class
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 

    return freq

#this function is used to label class 
#example if class is 0 then list looks like this
#[iparray,class0,1 for calss0,0]
#if its class6 then list looks like this
#[iparray,class6,0,1 for class 1]
def onehot(data):
  #apply one hot encoding on categorical features
  list1=[0,6]
  new_list=[]
  for i in range(0,len(data)):
    if data[i][1]==list1[0]:
      new_list.append([data[i][0],data[i][1],1,0])
   
    if data[i][1]==list1[1]:
      new_list.append([data[i][0],data[i][1],0,1]) 
  return new_list

#function is used to perform logistic regresion 
#for every iteration it updates weights bias and calculates dot produt 
#it returns bias,weights and p if its greater than 0.5 then assign one else o
def calclg(df,actual,w,bias,l_rate):
  z = np.dot(df,w)+bias
  z=1 / (1 + np.exp(-z))
  #print(z)
  bias+=l_rate*((actual-z)*z*(1-z))
  w+=l_rate*((actual-z)*z*(1 - z)*df)
 # print("weight up",w)
  #print("value",z)
  if z>0.5:
    p=1
  else:
    p=0
  return bias,w,p

#run this fucntion for selected epochs,it returns accuracy for each epoch and final accuracy
def lg_reg_c(data,list1,epochs):
  for j in range(epochs):
    output=[]
    counter=0
    w=np.linspace(-5,5,1024)
    #print(w)
    bias=0.001
    l_rate=0.01
    #shuffle data
    data=np.random.permutation(data)
    for i in range(0,len(data)):
      if data[i][1]==list1[0]:
        bias,w,z=calclg(data[i][0],data[i][2],w,bias,l_rate)
        if data[i][2]==z:
          counter+=1
    
      if data[i][1]==list1[1]:
        bias,w,z=calclg(data[i][0],data[i][3],w,bias,l_rate)    
        if data[i][3]==z:
          counter+=1

    accuracy=(counter/len(data))*100
    print("=====Epoch====={}====".format(j+1))
    print("Accuracy is {}".format(accuracy))

  return accuracy

def main():
  #get data for train
  data_train=preprocess_data("/devvoice/test")   
  #get data for test
  data_test=preprocess_data("/devvoice/training_validation")
  #perform one hot encoding for train and test data
  d_train=onehot(data_train)
  d_test=onehot(data_test)
  epochs=100
  list1=[0,6]
  st=time.time() #start time
  tr_accuracy=lg_reg_c(d_train,list1,epochs)
  print("====Training Accuracy====")
  print("Accuracy is {}".format(tr_accuracy))
  print("Time Elapsed for training {}".format(time.time()-st))
  st=time.time()
  tt_accuracy=lg_reg_c(d_test,list1,epochs)
  print("====Testing Accuracy====")
  print("Accuracy is {}".format(tt_accuracy))
  print("Time Elapsed for testing {}".format(time.time()-st))

main()
