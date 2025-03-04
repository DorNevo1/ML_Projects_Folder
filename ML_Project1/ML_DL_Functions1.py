import numpy as np

def LeastSquares(X,y):
  xT_x = X.T @ X
  xT_x_inv = np.linalg.inv(xT_x)
  theta = xT_x_inv @ X.T @ y
  return theta
  '''
    Calculates the Least squares solution to the problem X*theta=y using the least squares method
    :param X: numpy input matrix, size [N,m+1] (feature 0 is a column of 1 for bias)
    :param y: numpy input vector, size [N]
    :return theta = (Xt*X)^(-1) * Xt * y: numpy output vector, size [m+1]
    N is the number of samples and m is the number of features=28
  '''
  
  

def classification_accuracy(model,X,s):
  x_pred_new = model.predict(X)
  counter = 0
  for i in range(len(x_pred_new)):
    if x_pred_new[i] ==  s[i]:
      counter = counter + 1
  return counter/ len(x_pred_new)
  
  
  
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: numpy input matrix, size [N,m]
    :param s: numpy input vector of ground truth labels, size [N]
    :return: accuracy of the model = (correct classifications)/(total classifications) type float
    N is the number of samples and m is the number of features=28
  '''
  

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem. length 28
  '''
  return [-7.12910261e-02, -4.73665169e-03, -1.03978610e-02,  8.65626404e-03,
 -3.99926260e-02,  3.40262672e-03,  6.61865174e-02,  7.03472933e-05,
  2.03956248e-02, -3.19123317e-02,  4.35763050e-02,  8.35492096e-03,
  8.40613913e-02, 1.80697557e-01,  7.60750981e-01, 3.21130109e-02,
  1.44608543e-02, -1.02015057e-02,  1.96121548e-02, -5.94568044e-03,
  3.47496714e-02,  3.60067713e-02,  2.90424729e-03, -3.22749177e-02,
 -3.56838685e-02,  1.20871461e-02, -5.53989290e-03, -2.56530532e-02]


def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value. type float
  '''
  return 1.8989779321778998e-18

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of list of coefficiants for the classification problem.  length 28
  '''
  return [[-0.25220493, -0.29741098,  0.16431881, -0.14486237, -0.13429201, -0.32733827,
  -0.22000506,  0.0811542,  -0.11061615,  0.17684061, -0.30864931,  0.04196098,
  -0.10956331,  1.08033923,  2.86402312, -0.4430299,  -0.09542346, -0.0650778,
  -0.19082981, -0.19187785,  0.22886821,  0.25540807,  0.04251017, -0.11308635,
  -0.46784574,  0.09329152,  0.0533493,   0.39909522]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: list with the intercept value. length 1
  '''
  return [0.29508435]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem. length 2.
  '''
  return [0, 1]