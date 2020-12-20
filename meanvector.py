import numpy as  np
import re,time,os
import zipfile
import matplotlib.pyplot as plt
#to run the code give path details of datafiles
#extract the zip file 
zip_ref = zipfile.ZipFile("/content/drive/My Drive/dataset1.zip", 'r')
zip_ref.extractall("/devvoice") 
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

#preprocess the input data convert it to vector 
#the data is stored in respective class list that class0 in c0,class1 in c1,similarly for all classes
def preprocess_data(path):
  data_dir=os.listdir(path)
  #print("len data_dir",len(data_dir))
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
  for i in data_dir:
    if re.search('class_0',i,re.IGNORECASE):
      c0.append((img_vector(path,i)))
    
    if re.search('class_1',i,re.IGNORECASE):
      c1.append((img_vector(path,i)))
    
    if re.search('class_2',i,re.IGNORECASE):
      c2.append((img_vector(path,i)))
    
    if re.search('class_3',i,re.IGNORECASE):
      c3.append((img_vector(path,i)))
    
    if re.search('class_4',i,re.IGNORECASE):
      c4.append((img_vector(path,i)))
    
    if re.search('class_5',i,re.IGNORECASE):
      c5.append((img_vector(path,i)))
    
    if re.search('class_6',i,re.IGNORECASE):
      c6.append((img_vector(path,i)))
    
    if re.search('class_7',i,re.IGNORECASE):
      c7.append((img_vector(path,i)))
    
    if re.search('class_8',i,re.IGNORECASE):
      c8.append((img_vector(path,i)))
    
    if re.search('class_9',i,re.IGNORECASE):
      c9.append((img_vector(path,i)))

  #append mean in this list    
  mean_v=[]

  #creating a list so i can iterate and find mean of each class
  list1=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9]
  for i in list1:
    mean_v.append(np.mean(i,axis=0))  

  #to find the mean of whole data set I combine all classes into one array and find mean
  combine=(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9)
  total=np.vstack(combine)
  total=np.array(total)
  total_m=np.mean(total,axis=0)
  #append all mean in the mean_v list
  mean_v.append(total_m)
  return mean_v

#function to plot the graphs
def plot_graph(mean_v):
  #create a list of classes and list of color and these are passed to plot the graphs
  cla=["class0","class1","class2","class3","class4","class5","class6","class7","class8","class9","whole data"]
  color=['darkgreen','darkblue','yellow','orange','violet','red','navy','pink','gold','green','brown']
  x=list(range(0,1024)) #create component of the vector 
  #this plots all the graphs for class0 to 9 and whole data against the component vector on the horizontal axis
  for i in range(0,11):
    plt.title(r"{}".format(cla[i]),color="{}".format(color[i]),fontsize=16) #pass class name and color name
    plt.plot(x,mean_v[i],color="{}".format(color[i]))
    plt.xlabel('component of the vector 0 to 1023')
    plt.ylabel('mean of the component')
    plt.show()

#above are called in a sequence in main 
def main():
  #path of data and get mean and pass this mean in plot graph function
  mean_v=preprocess_data("/devvoice/training_validation")   
  plot_graph(mean_v)

main()









