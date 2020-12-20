import numpy as  np
import time
import matplotlib.pyplot as plt

def plot_graph():
  #plot multivariate gaussian distribution
  mean = [3,7]
  cov = [[6.9,-4.9],[-4.9,7]]
  x,y=np.random.multivariate_normal(mean, cov, 5000).T
  #multivariate gaussian distribution is ploted for 5k sample points with mean[3,7] and covaraince matrix [[6.9,-4.9],[-4.9,7]]
  plt.title("multivariate gaussian distribution")
  plt.scatter(x, y,color='cyan',marker="*",s=20)
  plt.axis("equal")
  plt.show()

  #another covariance matrix from the points drawn from the distribution, 
  # is decomposed using SVD the sample points are plotted in a uncorrelated (“circularized”) space.
  #new points drawn,decomposed and plotted in uncorrelated circularized space
  new=np.cov(x,y)
  u,s,vh = np.linalg.svd(new, full_matrices=True)
  #create a matrix with zeros
  sigma = np.zeros((2,2))
  #fill only diagonal elements with s values and remaining with zeros
  sigma[:2, :2] =np.diag(s)
  #invers the matrix
  si=np.linalg.inv(np.sqrt(np.matrix(sigma)))
  C = np.vstack((np.matrix(x),np.matrix(y)))
  #stack the matrix and x,y and then mutiply with the si and transpose of u
  output=si*u.T*C
  x1=list(output[0])
  y1=list(output[1])
  #plot x1 and y1 points 
  plt.title("sample points in a uncorrelated (“circularized”) space")
  plt.scatter(x1,y1,color='cyan',marker="*",s=20)
  plt.axis('equal')
  plt.show()

plot_graph()

