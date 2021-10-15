#extract zip file contents in one folder
import zipfile
import numpy as  np
import re,time,os

zip_ref = zipfile.ZipFile("/content/drive/My Drive/dataset1.zip", 'r')
zip_ref.extractall("/devvoice") 
zip_ref.close()

#this function converts images to 1-D vectors of of 32*32 i.e. 1024
#it takes data from files and converts to vectors for training and test data
#function take training dataset and test dataset path plus their individual file names
def img_vector(path,filename):
    return_vect = np.zeros((1,1024))
    fr = open(path+"/"+filename,"r")
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0,32*i+j] = int(line_str[j])

    return return_vect[0]

#to get the data from the folder
#to get data from folder assign classes i.e. class 0 as 0 class 1 as 1 so on for all clases
#this function returns processed data with class information added
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
    
#euclidian function to calulate distance
def euclidian(x,y):
    dist=[np.sqrt(sum(np.square(x-y)))][0]
    dist=round(dist,2)
    return dist

#function to sort the list in ascending order 
def sortvalues(newlist):
    #sort the list
    for i in range(0,len(newlist)):
        #print(newlist[i][2])
        for j in range(i + 1,len(newlist)):
            if(newlist[i][2] > newlist[j][2]):
                temp = newlist[i]
                newlist[i] = newlist[j]
                newlist[j] = temp
    #print("sorted list",newlist)
    return newlist

#function which counts frequency of class occuring in list and class predicted by classifier
def countfreq(my_list): 
    # count frequncy of class
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 

    return freq

#KNN classifier 
#calculate euclidian distance 
#for every of test set with respect to training set
#sort in ascending order
#for k values 
#select class that occurs maximum number of times as predicted class     
#for k equal to 4 class1 occurs thrice and class0 occurs once then class one is selected
def KNN(k,train_set,test_set):
    #for every test and train set
    counter=0
    for i, val in enumerate(test_set):
      newlist=[] #calculate distance with every training se
      for j,val1 in enumerate(train_set):
        #print("value",val) 
        newlist.append([val1[0],val1[1],euclidian(val1[0],val[0])])
        #sort the list in ascending order
      nl=sortvalues(newlist)  
      #print("length list",nl)  
      nl=nl[:k]
      #print("newlist",newlist)
      c=[]#to find the predicted class
      for p,val2 in enumerate(nl):
        c.append(val2[1])
      
      c=countfreq(c)
      predicted=max(c,key=c.get)
      if val[1]==predicted:
        counter+=1

    accuracy=float(counter/len(test_set))*100
    return accuracy

#5 fold cross validation is applied 
# in this there are total 1934 rows in training dataset
#so 1934 rows are divided in to 5 folds of 386
#for first iteration first 386 rows is test and remaining is training
#so onwards for second iteration next 386 becomes test and other becomes training 
#takes next fold and then finally last fold becomes test and other becomes training
#before apply 5 fold we shuffle the data set so we get mixed values
#running for different values of k to find the best K
def crossval(data):
  best_k=[]#to find best k after iterating for k=1 to 11 and applying 5 folds at each step
  for k in range(0,11):
    #apply 5 fold cross validation for k=1...to...11 to find best k
    a=[]#accuracy
    #shuffle data
    data=np.random.permutation(data)
    for i in range(0,5):
      #split array into number of folds
      f=np.array_split(data,5)
      #select each fold ieratively for test and train sets
      test_set=f[0]
      f=np.delete(f,0)
      train_set=np.concatenate((f), axis=0)
      #print("train_set",train_set)  
      accuracy=KNN(k+1,train_set,test_set)
      #here printing accuracy of each fold for value of k 
      print("Accuracy for k={} and fold={} out of 5".format(k+1,i+1))
      print("Accuracy is ",accuracy)
      a.append(accuracy)

    #here printing final accuracy after 5 folds for the given k value
    print("=====Accuracy after applying 5 fold cross validation for given k value=====")
    print("Accuracy for k={}".format(k+1))
    acc=float(sum(a)/len(a))
    print("Accuracy is ",acc)
    best_k.append(acc)

  return best_k
         
def main():
  data_train=preprocess_data("/devvoice/training_validation")   
  data_test=preprocess_data("/devvoice/test")
  st=time.time() #start time
  val=crossval(data_train)
  #total time required to apply cross validation
  print("Time Elapsed applying cv for k=1 to11 {}".format(time.time()-st))
  print("value",val)
  #know we find the best k
  #below give lambda function returns the best accuracy
  #from the list of accuracy for k equal to 1 to11
  #then we find average of this accuracy
  #based on the best value we find which value is close by to our best value
  #we check for best value in our list and find its index position
  #these are the values in my list when i did cross validation for my k values 
  #[97.1576227390181,97.41602067183463,96.64082687338501,96.64082687338501,98.19121447028424,95.8656330749354,95.60723514211887,96.38242894056847,97.41602067183463,96.12403100775194,94.57364341085271]
  difference_function = lambda list_value : abs(val - sum(val)/len(val))
  closest_value = min(val, key=difference_function)
  print("closest_value",closest_value)
  #i get closest_value this is 96.64082687338501
  #then i find the index position for this value 96.64082687338501
  #so i select index position 2 which occurs first while comparing so my index position 2 which represents value for k=3 so my best is k=3
  pos=val.index(closest_value)  
  k=pos+1
  print("=====Best K is====={}".format(k))
  st=time.time() #start tim
  pred=KNN(k,data_train,data_test)
  print("Accuracy on Testing set is {}".format(pred))
  print("Time Elapsed Testing {}".format(time.time()-st))

main()

