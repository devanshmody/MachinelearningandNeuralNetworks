import numpy as np
import matplotlib.pyplot as plt

#gradient_1 function is for the part 1 of the problem 6
def gradient_1():
  b=4.5 #initial values of beta 
  l_rate=0.001 #learning rate 
  d=1 # for difference
  p=0.00001 #precision
  max_limit=1000 #max iteration limit
  limit=900 #loop over till 900 
  x=np.linspace(-5,5,1000) #generate thousand random value between -5 to 5
  y=x**2 #our function
  #plot the function
  plt.plot(y,color="DARKBLUE") 
  plt.title(r'function f(β) = β^2',color="DARKGREEN",fontsize=16)
  plt.xlabel("plotting for the function")
  plt.show()
  
  #iterate for graident descent till a certain limit 
  for i in range(limit):
    if d>p and i <max_limit:
      output=plt.scatter(i,b,marker='*',s=20,cmap='viridis') 
      b_old=b #copying beta value to beta old 
      b=b-l_rate*(2*b_old) #update values
      d=abs(b-b_old) #calculate difference

  #plot the gradient
  plt.title(r"Gradient of function f(β) = β^2",color="DARKGREEN",fontsize=16)
  plt.xlabel('Beta iterations')
  plt.ylabel('Beta Values')
  plt.legend([output],["beta points"])
  plt.show()

# for part 1 of the problem 6
gradient_1()

#gradient_2 function is for the part 2 of the problem 6
def gradient_2():
  b = 2.4 #initial values of beta
  l_rate = 0.001 #learning rate
  d = 1 #difference
  p = 0.00001 #precision
  limit =900  #loop over till 900 
  max_limit=1000 #max iteration limit
 
  x = np.linspace(0.5,2.5,1000) #genrate thousand random values between 0.5 and 2.5
  y= [(np.sin(10* np.pi*x[i])/(2*x[i])) - (x[i] -1)**4 for i in range(len(x))] #our function
  #plot the function
  plt.plot(x,y,color="DARKBLUE")
  plt.title(r'function sin(10πβ)/2β+ (β − 1)^4', fontsize=16, color='DARKGREEN')
  plt.xlabel("plotting for the function")
  plt.show()    

  #iterate for graident descent till a certain limit
  for i in range(limit):
    if d>p and i<max_limit: 
      output = plt.scatter(i,b,cmap='viridis',marker="*",s=20)
      b_old = b  #copying beta value to beta old 
      b = b - l_rate * (4*(b-1)**3 + 5*np.pi*np.cos(10*np.pi*b)/b - np.sin(10*np.pi*b)/(2*b**2)) #update values
      d = abs(b - b_old) #calculate difference
  
  #plot the gradient
  plt.title(r'Gardient of function f(β) = sin(10πβ)/2β+ (β − 1)^4',color="DARKGREEN",fontsize=16)
  plt.xlabel("Beta iterations")
  plt.ylabel("Beta values")
  plt.legend([output],['beta points'])
  plt.show()

# for part 2 of the problem 6
gradient_2()

