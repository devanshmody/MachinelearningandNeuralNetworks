import zipfile
import numpy as  np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re,time,os
#to run the code give path details of datafiles in code 
#extract the zip file 
zip_ref = zipfile.ZipFile("/content/drive/My Drive/dataset1.zip", 'r') #give path details here
zip_ref.extractall("/devvoice")   #data set gets extracted in this folder
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

#plot the graph of principle components
#this function takes the input and arranges them in classwise in list c0 to c9 for plotting it in a 3d space
def plot_projection(df):
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
  #so classwise data is stored in the list ravel is used to flatten the input array
  for val in df:
    if val[1]==0:
      c0.append(val[0].ravel())
    if val[1]==1:
      c1.append(val[0].ravel())
    if val[1]==2:
      c2.append(val[0].ravel())
    if val[1]==3:
      c3.append(val[0].ravel())
    if val[1]==4:
      c4.append(val[0].ravel())
    if val[1]==5:
      c5.append(val[0].ravel())
    if val[1]==6:
      c6.append(val[0].ravel())
    if val[1]==7:
      c7.append(val[0].ravel())
    if val[1]==8:
      c8.append(val[0].ravel())
    if val[1]==9:
      c9.append(val[0].ravel())
  #convert back to array for plotting
  c0=np.array(c0)
  c1=np.array(c1)
  c2=np.array(c2)
  c3=np.array(c3)
  c4=np.array(c4)
  c5=np.array(c5)
  c6=np.array(c6)
  c7=np.array(c7)
  c8=np.array(c8)
  c9=np.array(c9)  
  #create a list of colors and labels for the classes
  color=["b","g","r","c","m","y","orange","navy","pink","violet"]
  labels=["class0","class1","class2","class3","class4","class5","class6","class7","class8","class9"]
  clas=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9]

  # Creating figure
  fig = plt.figure(figsize = (16, 9))
  ax = plt.axes(projection ="3d")
   
  # Add x, y gridlines 
  ax.grid(b=True,color ='grey',linestyle ='-.', linewidth = 0.3,alpha = 0.2)

  #a loop is called and class labels and color information and class information is passed and plotted
  for i in range(0,len(clas)):
    sctt = ax.scatter3D(clas[i][:,:1],clas[i][:,1:2],clas[i][:,2:3],c=color[i],marker ='*',label=labels[i])
  
  plt.title("Projection of principle components")
  ax.set_xlabel('X-axis', fontweight ='bold') 
  ax.set_ylabel('Y-axis', fontweight ='bold') 
  ax.set_zlabel('Z-axis', fontweight ='bold')
  plt.legend(loc='upper left')
  fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
  # show plot
  plt.show()

#get the data preproces the data convert image to vector and one hot encode the data based on the class labels
#that is for class0 this is the representation 1,0,0,0,0,0,0,0,0,0 .... class 9 0,0,0,0,0,0,0,0,0,1
#so data appended in list trv_data looks like this for individal classes
#array[1024 values],class label 0 to 9 for individual classes,one hot encoded information for 0 to 9 classes
def preprocess_data(path):
  data_dir=os.listdir(path)
  #print("len data_dir",len(data_dir))
  trv_data=[]
  for i in data_dir:
    if re.search('class_0',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),0,1,0,0,0,0,0,0,0,0,0))
    
    if re.search('class_1',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),1,0,1,0,0,0,0,0,0,0,0))
    
    if re.search('class_2',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),2,0,0,1,0,0,0,0,0,0,0))
    
    if re.search('class_3',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),3,0,0,0,1,0,0,0,0,0,0))
    
    if re.search('class_4',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),4,0,0,0,0,1,0,0,0,0,0))
    
    if re.search('class_5',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),5,0,0,0,0,0,1,0,0,0,0))
    
    if re.search('class_6',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),6,0,0,0,0,0,0,1,0,0,0))
    
    if re.search('class_7',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),7,0,0,0,0,0,0,0,1,0,0))
    
    if re.search('class_8',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),8,0,0,0,0,0,0,0,0,1,0))
    
    if re.search('class_9',i,re.IGNORECASE):
        trv_data.append((img_vector(path,i),9,0,0,0,0,0,0,0,0,0,1))
      
  #print("len data",len(trv_data)) 
  return trv_data

#get reduce dimension using svd and principle components with maximum information or variance threshold is 95%
def svd_pca(data,info):
  total=[]
  for val in data:
    #print(val[1])
    total.append(val[0])
  #convert to array
  X=np.array(total)
  #calculate mean    
  m=np.mean(X,axis=0)  
  #subtract mean from input
  d=X-m 
  #calculat covariance 
  covm=np.cov(d,rowvar=0)
  #decompose using SVD
  u,s,vh = np.linalg.svd(covm, full_matrices=True) 
  #find maximum variance or maximum information threshold is 0.95 or 95%
  variance=np.cumsum(s)/np.sum(s) 
  #get number of resulting principle components
  K = np.argmax(variance>0.95)+1
  print("Number of components resulting from the dimensionality reduction are {}".format(K))
  final=np.matmul(d,u[:,:K])
  #print(np.shape(final))
  final_df=[]
  #add class information to reduced dimensions and append them in the final output
  #ravel is used to flatten the matrix
  for i in range(0,len(data)):
    final_df.append([final[i].ravel(),data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7],data[i][8],data[i][9],data[i][10],data[i][11]])

  print("Projection of points")
  #plot the projection of points in 3D space
  plot_projection(final_df)
  return final_df,u[:,:K],K

#calculate logistic regression
def calclg(df,actual,w,bias,l_rate):
  z = np.dot(df,w)+bias
  z=1 / (1 + np.exp(-z))
  #print(z)
  bias+=l_rate*((actual-z)*z*(1-z))
  w+=l_rate*((actual-z)*z*(1 - z)*df)
  #print("weight up",w)
  #print("value",z)
  if z>0.5:
    p=1
  else:
    p=0
  
  return bias,w,p

#run this fucntion for selected epochs,it returns accuracy for each epoch and final accuracy
def lg_reg_c(data,list1,epochs,d):
  #generate intial weights randomly
  #here d is the number of features without pca we have 1024 so 1024 weights will be generated
  #with pca dimension is reduced so according depending on the dimesion values weights will be generated
  #if we are getting 467 components then weights will be generated for 467 features 
  w=np.linspace(-5,5,d) 
  #print(w)
  bias=0.9
  l_rate=0.01
  #shuffle data   
  data=np.random.permutation(data)
  #for the number of given epochs call logistic regresion function update weights and compare actual with predicted value and return accuracy and final accuracy after completion of all the epochs
  for j in range(epochs):
    counter=0
    for val in data:
      if val[1]==list1[0]:       
        bias,w,z=calclg(val[0],val[2],w,bias,l_rate)
        if val[2]==z:
          counter+=1
    
      if val[1]==list1[1]:
        bias,w,z=calclg(val[0],val[3],w,bias,l_rate)    
        if val[3]==z:
          counter+=1
      
      if val[1]==list1[2]:
        bias,w,z=calclg(val[0],val[4],w,bias,l_rate)    
        if val[4]==z:
          counter+=1

      if val[1]==list1[3]:
        bias,w,z=calclg(val[0],val[5],w,bias,l_rate)    
        if val[5]==z:
          counter+=1

      if val[1]==list1[4]:
        bias,w,z=calclg(val[0],val[6],w,bias,l_rate)    
        if val[6]==z:
          counter+=1

      if val[1]==list1[5]:
        bias,w,z=calclg(val[0],val[7],w,bias,l_rate)    
        if val[7]==z:
          counter+=1

      if val[1]==list1[6]:
        bias,w,z=calclg(val[0],val[8],w,bias,l_rate)    
        if val[8]==z:
          counter+=1

      if val[1]==list1[7]:
        bias,w,z=calclg(val[0],val[9],w,bias,l_rate)    
        if val[9]==z:
          counter+=1

      if val[1]==list1[8]:
        bias,w,z=calclg(val[0],val[10],w,bias,l_rate)    
        if val[10]==z:
          counter+=1
      
      if val[1]==list1[9]:
        bias,w,z=calclg(val[0],val[11],w,bias,l_rate)    
        if val[11]==z:
          counter+=1

    accuracy=(counter/len(data))*100
    #print("=====Epoch====={}====".format(j+1))
    #print("Accuracy is {}".format(accuracy))

  return accuracy

#Logistic regression with PCA
def logistic_pca(data_train,data_test,epochs,list1):
  st=time.time() #start time
  #svd_pca function reduces dimesion,plot the projection 
  #data_train_pca is data with reduced dimension,U are the selected princile components >0.95 and K is the number of principle components
  data_train_pca,U,K=svd_pca(data_train,"Training")
  trl_accuracy=lg_reg_c(data_train_pca,list1,epochs,K)
  print("====Training Accuracy With PCA====")
  print("Accuracy is {}".format(trl_accuracy))
  print("Time Elapsed for training {}".format(time.time()-st))

  #use the U output received to reduce the dimensions of the test data and apply logistic regression on the test data
  total=[]
  for val in data_test:
    #print(val[1])
    total.append(val[0])

  #mutiplt test data with U the selected principle components 
  #ravel is used to flatten the matrix
  final=np.matmul(total,U)  
  final_df=[]
  for i in range(0,len(data_test)):
      final_df.append([final[i].ravel(),data_test[i][1],data_test[i][2],data_test[i][3],data_test[i][4],data_test[i][5],data_test[i][6],data_test[i][7],data_test[i][8],data_test[i][9],data_test[i][10],data_test[i][11]])

  st=time.time()
  ttl_accuracy=lg_reg_c(final_df,list1,epochs,K)
  print("====Testing Accuracy with PCA====")
  print("Accuracy is {}".format(ttl_accuracy))
  print("Time Elapsed for testing {}".format(time.time()-st))

def main():
  #get data for train provide the path here
  data_train=preprocess_data("/devvoice/training_validation")   
  #get data for test provide the path here
  data_test=preprocess_data("/devvoice/test")
  epochs=1000 #run for the 1k epochs
  list1=[0,1,2,3,4,5,6,7,8,9]
  print("length training data {} and testing data {}".format(len(data_train),len(data_test)))

  #call the logistic regression function with PCA for train and test data
  logistic_pca(data_train,data_test,epochs,list1)

  #Logistic regression without PCA
  st=time.time() #start time
  #here dn contains the shape of input array, that is the shape is 1024
  dn=np.shape(data_train[0][0])[0] 
  tr_accuracy=lg_reg_c(data_train,list1,epochs,dn)
  print("====Training Accuracy Without PCA====")
  print("Accuracy is {}".format(tr_accuracy))
  print("Time Elapsed for training {}".format(time.time()-st))

  st=time.time()
  tt_accuracy=lg_reg_c(data_test,list1,epochs,dn)
  print("====Testing Accuracy without PCA====")
  print("Accuracy is {}".format(tt_accuracy))
  print("Time Elapsed for testing {}".format(time.time()-st))


main()

