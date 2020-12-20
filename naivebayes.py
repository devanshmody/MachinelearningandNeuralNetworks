import zipfile
import numpy as  np
import re,time,os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wl
from nltk.stem import PorterStemmer 
ps = PorterStemmer() # create stemmer object
lmtzr = nltk.WordNetLemmatizer() #create lemmatizer object
#extracted the files in devvoice folder of my drive
zip_ref = zipfile.ZipFile("/content/drive/My Drive/dataset2.zip", 'r')
zip_ref.extractall("/devvoice")
zip_ref.close()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

#this function does all the work of cleaning datasets
def processing(path,filename):
  #print("path",path)
  #here path is trainig or testing drictory path depending values passed to path
  #file name is name of files in training and testing directory
  f=open(path+"/"+filename,"rb")
  p=f.readlines()
  f.close()
  #open files and do the cleaning
  #first convert it to lower case
  p=re.sub('[A-Z]+', lambda m: m.group(0).lower(),str(p)) 
  #remove email id's from messages
  p=re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+",'',p) 
  #remove everything before the word lines as they are the headers
  p=re.sub(r'^.*lines','',p)
  #remove single characters
  p=re.sub(r'\b\w\b', '',p)
  #replace \\n with blank 
  p=p.replace('\\n','')
  #replace quotes with blank
  p=p.replace("'",' ')
  #there few delimeters like these ‘---’ ,’-- ‘,’--  ‘,thanx,thanks 
  #so remove those delimeters from the message these delimiters
  #indicate end of message and there are signatures at the 
  #end of the message so remove it and keep the first part of  the message that is the message body
  p=p.split(r'---')[0]
  p=p.split(r"--")[0]
  p=p.split(r'-- ')[0]
  p=p.split('thanks')[0]
  p=p.split('thanx')[0]
  #if there are still any headers remaining the find those and remove them
  t=re.findall(r'[\w\s.-]+:[\w\s.-]+',p) #list containing additional header information

  #apply stemming and lemmetization
  #stemming and lemmatization is to reduce inflectional forms and sometimes 
  #derivationally related forms of a word to a common base form
  #example words 'am','are','is'  becomes be 
  # words 'car','cars','car's,'cars' becomes 'car'

  #the p list contains the whole message 
  #the t list contains additional headers which needs to be removed from p list
  # so apply tokenize,lemmetize,steming,stemming on both list
  #the reomve stop words and alphanumeric words and then finally 
  #remove all adiitional headers and generate a new list that is t2
  #then using the list which contains additional headers 
  #i remove the additional header from the original list

  to=word_tokenize(p) #tokenize
  to = [lmtzr.lemmatize(token) for token in to] #lemmetize the words
  to = [ps.stem(token) for token in to] #stemm the words
  to = [w for w in to if not w in stop_words] #remove stop words
  to = [w for w in to if w.isalpha()] #remove alphanumeric words

  t1=word_tokenize(str(t))
  t1 = [lmtzr.lemmatize(token) for token in t1]
  t1 = [ps.stem(token) for token in t1]
  t1 = [w for w in t1 if not w in stop_words]
  t1 = [w for w in t1 if w.isalpha()]

  #rmove headers from main body of the message
  #by taking those elements from to list which are not in t1 list
  #this way all aditional headers are removed
  t2 = [x for x in to if x not in t1] 
  t2 = [t2 for t2 in t2 if  not t2.isdigit()]
  t2=' '.join(t2)
  #remove punctuation
  t2 = re.sub(r'[^\w\s]',' ', t2) 
  t2=word_tokenize(t2)
  t2 = [w for w in t2 if not w in stop_words]
  t2 = [t2 for t2 in t2 if  not t2.isdigit()]
  #remove blank spcaes
  t2 = [i for i in t2 if i] 
  #select words with length greater than 2 so that words like a,aa etc get removed
  t2 = [i for i in t2 if len(i)>2] 

  return t2

#this function is used to count the frequency of the words in my vocabulary of words
def countfreq(my_list): 
  # count frequncy of class
  freq = {} 
  for items in my_list:
    freq[items] = my_list.count(items) 
        
  return freq

#get data for training
#here all the data is collected for train set 
#train set contains “sic.electronics” and “comp.sys.ibm.pc.hardware”.
#data is fetched for both vocabulary is created and frequency of words 
#is counted and returned for “sic.electronics” and “comp.sys.ibm.pc.hardware” train set
def getdata_train(path):
  data_dir=os.listdir("{}".format(path))
  vocab=[]
  #to get the data from the folder
  #def preprocess_data gets all the data
  for i in data_dir:
    #print(i)
    vocab.extend(processing(path,i))

  #to count frequency of words
  feq_set=countfreq(vocab) 
  length=len(data_dir)
  return feq_set,length

#get data for testing
#here all the data is collected for test set 
#test set contains “sic.electronics” and “comp.sys.ibm.pc.hardware”.
#token of words is returned for “sic.electronics” and “comp.sys.ibm.pc.hardware” test set
#append class labels also 
#i.e ibm if it belongs to comp.sys.ibm.pc.hardware 
# sci if it belongs to sic.electronics
def getdata_test(path,label):
  data_dir=os.listdir("{}".format(path))
  vocab=[]
  labels=[]
  #to get the data from the folder
  #def preprocess_data gets all the data
  for i in data_dir:
    #print(i)
    vocab.append(processing(path,i))
    labels.append(label)
    
  return vocab,labels

#this function calculates probablity of given words
#belongs to IBM or SCI class
#p(given words | IBM)
#p(given words | SCI)
#here first it runs for (given words|ibm) then (given words|sci) 
def g_w_ibm_sci(df,train_d,l,pb):
  data=[]
  for i in range(0,len(df)):
    # print(test_set[0][i])
    for key,val in train_d.items():
      #print(val)
      # print("value",val[0])
      if key==df[i]:
#        print("match",val,df[i])
        data.extend([np.log((val)/(l))])
  p=np.exp(sum(data)+pb)  
  return p

#this function calls g_w_ibm_sci
#to get the probablities for given belonging to ibm or sci
def f_prob(train_ibm,tribm_l,train_sci,trsci_l,p_ibm,p_sci,test_c):
  pibm=[]
  psci=[]
  for i in range(0,len(test_c)):
#    print(test_c[i])
    pibm.append(g_w_ibm_sci(test_c[i],train_ibm,tribm_l,p_ibm))
    psci.append(g_w_ibm_sci(test_c[i],train_sci,trsci_l,p_sci))
  
  return pibm,psci

#all the above functions are called in a squence and final results are printed
def main():
  #get training vocabulary of words for ibm with frequency occurence of each word
  train_ibm,tribm_l=getdata_train("/devvoice/dataset2/train/comp.sys.ibm.pc.hardware")
  #get training  vocabulary of words for sci with frequency occurence of each word
  train_sci,trsci_l=getdata_train("/devvoice/dataset2/train/sci.electronics")
  #get test dataset for ibm
  test_ibm,tibm_l=getdata_test("/devvoice/dataset2/test/comp.sys.ibm.pc.hardware","IBM")
  #get test dataset for sci
  test_sci,tsci_l=getdata_test("/devvoice/dataset2/test/sci.electronics","SCI")

  #find total of ibm and sci classes
  total_train=tribm_l+trsci_l
  print("length of training data")
  print("length of IBM={} SCI={} and total length {}".format(tribm_l,trsci_l,total_train))
  print("total number of tokens in our training set IBM={} SCI={}".format(len(train_ibm),len(train_sci)))

  p_ibm=np.log(tribm_l/total_train) #p(ibm/total)
  p_sci=np.log(trsci_l/total_train) #p(sci/total)
  print("length of testing data")
  print("length of IBM={} SCI={} and total length {}".format(len(test_ibm),len(test_sci),len(test_ibm)+len(test_sci)))
  #print("probablity p(ibm)={} and p(sci)={}  ".format(p_ibm,p_sci))

  test_c=test_ibm+test_sci #combine test set
  c_label=tibm_l+tsci_l #combine labels

  #call the naive bayes classifier
  pibm,psci=f_prob(train_ibm,tribm_l,train_sci,trsci_l,p_ibm,p_sci,test_c)

  #find probablity which is greater than other and assign correspoind class to it
  pred=[]
  for i in range(0,len(c_label)):
    if pibm[i]>psci[i]:
      pred.append("IBM")
    else:
      pred.append("SCI")
  
  #calculate accuracy 
  counter=0  
  for j in range(0,len(c_label)):
    if c_label[j]==pred[j]:
      #print("matched",c_label[j],pred[j])
      counter+=1

  accuracy=(counter/len(c_label))*100
  print("=====Final Performance=====")
  print("Accuracy of the classifier is {}".format(accuracy))

st=time.time()
main()
print("Time Elapsed {}".format(time.time()-st))

