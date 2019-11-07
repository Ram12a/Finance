import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

class PlotTools():
    
    def Histogram(self,X,y,c,cmap,size):
        plt.scatter(X,y,c,cmap,size)
        return None
    
class Features():
    
    def expandQuadratics(self,X):
        """
        Adds quadratic features. 
        This expansion allows your linear model to make non-linear separation.

        For each sample (row in matrix), compute an expanded row:
        [feature0, feature1, feature0^2, feature1^2, feature1*feature2, 1]

        :param X: matrix of features, shape [n_samples,2]
        :returns: expanded features of shape [n_samples,6]
        """
        X_expanded = np.zeros((X.shape[0], 6))

        X_expanded[:,0] = X[:,0]
        X_expanded[:,1] = X[:,1]
        X_expanded[:,2] = np.power(X[:,0],2)
        X_expanded[:,3] = np.power(X[:,1],2)
        X_expanded[:,4] = X[:,0]*X[:,1]
        X_expanded[:,5] = 1
        return X_expanded
    
class OwnModels(Features):
    def OwnLogisticProbabilistic(self,X,w):
        """
        Given input features and weights
        return predicted probabilities of y==1 given x, P(y=1|x), see description above

        Don't forget to use expand(X) function (where necessary) in this and subsequent functions.

        :param X: feature matrix X of shape [n_samples,6] (expanded)
        :param w: weight vector w of shape [6] for each of the expanded features
        :returns: an array of predicted probabilities in [0,1] interval.
        """
        p = 1./(1+np.exp(-np.dot(X,w)))
        return p
    
    def OwnLogisticLoss(self,X, y, w):
        """
        Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
        and weight vector w [6], compute scalar loss function using formula above.
        """
        L = -(1/y.shape[0])*(np.dot(y,np.log(self.OwnLogisticProbabilistic(X, w))) + np.dot((1-y),np.log(1-self.OwnLogisticProbabilistic(X, w))))
        return L
    
    def OwnLogisticGrad(self,X, y, w):
        """
        Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
        and weight vector w [6], compute vector [6] of derivatives of L over each weights.
        """
        Grad = (1./X.shape[0])*np.dot(X.transpose(),(self.OwnLogisticProbabilistic(X,w)-y))
        return Grad
    
    def VisualizeGradientWeight(self,X,y,w,history,h=0.01):
        """draws classifier prediction with matplotlib magic"""
        h = h
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.OwnLogisticProbabilistic(self.expandQuadratics(np.c_[xx.ravel(), yy.ravel()]), w)
        Z = Z.reshape(xx.shape)
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.subplot(1, 2, 2)
        plt.plot(history)
        plt.grid()
        ymin, ymax = plt.ylim()
        plt.ylim(0, ymax)
        display.clear_output(wait=True)
        plt.show()
    
    def MiniBatchGD(self,X,y,w,n_iter,batchSize,lr=0.1,Visualize=True):
        loss = np.zeros(n_iter)
        if Visualize:
            for i in range(n_iter):
                ind = np.random.choice(X.shape[0],batchSize)
                loss[i] = self.OwnLogisticLoss(X,y,w)
                if i % 10 == 0:
                    self.VisualizeGradientWeight(X[ind,:],y[ind],w,loss)
                w = w - lr*self.OwnLogisticGrad(X[ind,:],y[ind],w)
        else:
            for i in range(n_iter):
                ind = np.random.choice(X.shape[0],batchSize)
                w = w - lr*self.OwnLogisticGrad(X[ind,:],y[ind],w)
        return w
            
    def MiniBatchGDMomentum(self,X,y,w,n_iter,batchSize,lr=0.1,Visualize=True):
        loss = np.zeros(n_iter)
        if Visualize:
            for i in range(n_iter):
                ind = np.random.choice(X.shape[0],batchSize)
                loss[i] = self.OwnLogisticLoss(X,y,w)
                if i % 10 == 0:Feat = Features()
                    self.VisualizeGradientWeight(X[ind,:],y[ind],w,loss)
                nu = alpha*nu + eta*compute_grad(X[ind,:],y[ind],w)
                w = w - nu
        else:
            for i in range(n_iter):
                ind = np.random.choice(X.shape[0],batchSize)
                nu = alpha*nu + eta*compute_grad(X[ind,:],y[ind],w)
        return w
        
        

        