import zipfile
import numpy as  np
import re,time,os
#to run the code give path details of datafiles in code 
#extract the zip file 
zip_ref = zipfile.ZipFile("/content/drive/My Drive/dataset1.zip", 'r') #give path details here 
zip_ref.extractall("/devvoice")  #data set gets extracted in this folder
zip_ref.close()

#this fucntion is used to convert the image to vector
#it converts a 32*32 to a 1024 vector or array
def img_vector(path,filename):
  return_vect = np.zeros((1,1024))
  fr = open(path+"/"+filename,"r")
  for i in range(32):
    line_str = fr.readline()
    for j in range(32):
      return_vect[0,32*i+j] = int(line_str[j])

  return return_vect[0]

# this fucntion is used to fetch the data convert image to a vector and assigne a class number to it
#if its class0 then assign class0 label 0 and so on for classes 1 for class1 ....9 for class9
def preprocess_data(path):
  data_dir=os.listdir(path)
  #print("len data_dir",len(data_dir))
  
  trv_data=[]
  for i in data_dir:
    if re.search('class_0',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),0))
    
    if re.search('class_1',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),1))
    
    if re.search('class_2',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),2))
    
    if re.search('class_3',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),3))
    
    if re.search('class_4',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),4))
    
    if re.search('class_5',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),5))
    
    if re.search('class_6',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),6))
    
    if re.search('class_7',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),7))
    
    if re.search('class_8',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),8))
    
    if re.search('class_9',i,re.IGNORECASE):
      trv_data.append((img_vector(path,i),9))
      
    #print("len data",len(trv_data)) 
        
  return trv_data


#thsi fucntion does all the calculation part it calculates mean,covariance,priors
#for Qda covariance is calulated class wise so c0 to c9 are lists for classes 0 to 9
#then operations are performed on the c0 to c9 lists of arrays
def mean_vec(data):
  #create empty lists
  c0=[]
  c1=[]
  c2=[]
  c3=[]
  c4=[]
  c5=[]
  c6=[]
  c7=[]
  c8=[]
  c9=[]
  for key,val in data:
    if val==0:
      c0.append(key)
 
    if val==1:
      c1.append(key)
  
    if val==2:
      c2.append(key)
  
    if val==3:
      c3.append(key)
  
    if val==4:
      c4.append(key)
  
    if val==5:
      c5.append(key)
  
    if val==6:
      c6.append(key)
  
    if val==7:
      c7.append(key)

    if val==8:
      c8.append(key)
  
    if val==9:
      c9.append(key)

  #cobine all class information into a single list for further calculations
  list1=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9]
  
  #create a list to append the final output  of all the calculations
  final_o=[]
  #label is used to add the class information that is label 0 means class0,1 means class1 and so on till 9 for class9
  label=0
  for i in list1: 
    #convert to array
    i=np.array(i)  
    #calculate the mean
    m=np.mean(i,axis=0)
    #subtract mean from input
    d=i-m 
    #calculate covariance matrix
    covm=np.cov(d,rowvar=0) 
    #print(np.shape(np.array(covm)))
    u,s,vh = np.linalg.svd(covm, full_matrices=True) 
    counter=0
    for j in range(0,len(s)):
      #if there are any infinite values positive or negative finite those values and then take count of only those  values greater than zero 
      if np.nan_to_num(s[j],copy=True,nan=0.0,posinf=True,neginf=True)>0.0:
        counter+=1
    sigma=np.zeros((1024,1024))
    sigma[:1024,:1024]=np.diag(s)
    si=np.linalg.inv(np.sqrt(np.array(sigma)))
    #based on the counter values consturct back the covariance matrix
    cov=u[0:counter,0:counter]*sigma[0:counter,0:counter]*vh[0:counter,0:counter]
    #print(np.shape(u),np.shape(s),np.shape(vh))
    final_o.append([m,np.log(len(i)/len(data)),si,u,label,np.log(np.linalg.det(cov))]) #mean,theta,si,u,class label and covariance matrix of each class
    label+=1               
  return final_o

#this function is the QDA classifier
def QDA(df,f_d): 
  correct=0
  for key,val in df:
    max1=[]
    #f_d[0]contains mean,f_d[1]contains theta or priors,f_d[2] contains sigma,f_d[3] contains u vector,
    #f_d[4] contains class label and f_d[5] contains covaraince matrix
    for i in range(0,len(f_d[0])): 
      a1=(f_d[i][2]*f_d[i][3].T*key)-(f_d[i][2]*f_d[i][3].T*f_d[i][0])
      b1=np.transpose(a1)
      #convert to finite values
      det=np.nan_to_num(f_d[i][5],copy=True,nan=0.0,posinf=True,neginf=True)
      c=-0.5*det-0.5*(b1*a1)+f_d[i][1]
      #find the maximum and append in list max1
      max1.append(np.max(c)) 
    #from the list of maximum values get the max value and index of max value
    max_value = np.max(max1)
    max_index = max1.index(max_value)
    #calculate accuracy
    #check class label is equal to index value of maximum, here index value indicates the class label predicted by classifier
    if val==max_index:
      correct+=1
      accuracy=(correct/len(df))
      print("Accuracy is {}".format(accuracy))
  
  return accuracy

#all the above function are called in a sequence 
def main():
  #get train and test data provide the paths for training and testing
  data_train=preprocess_data("/devvoice/training_validation")   
  data_test=preprocess_data("/devvoice/test")
  #shuffle the data
  data_train=np.random.permutation(data_train)
  #call mean_vec fucntion
  final=mean_vec(data_train)
  st=time.time() #start time
  tr_accuracy=QDA(data_train,final)
  print("====Training Accuracy====")
  print("Accuracy is {}".format(tr_accuracy))
  print("Time Elapsed for training {}".format(time.time()-st))
  st=time.time()
  tt_accuracy=QDA(data_test,final)
  print("====Testing Accuracy====")
  print("Accuracy is {}".format(tt_accuracy))
  print("Time Elapsed for testing {}".format(time.time()-st))

main()

