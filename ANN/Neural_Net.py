import numpy as np
import random

class NeuralNet:
    
    """
    A neural network of size given by the required input size_net. 
    
    Can train the network with either batch gradient descent or minibatch 
    stochastic gradient descent. 
            
    Parameters
    ----------
    size_net: array-like 
            first element is how many inputs exist and then how many hidden 
            neurons each layer has last is then the output layer.
    
    Functions            
    ---------
    train_network_batch
    train_network_SGD
    
    """
    
    def __init__(self,size_net):
        self.size_net = size_net
        self.num_lay = len(size_net)
        self.bias = [np.random.randn(b,1) for b in size_net[1:]]
        self.weight = [np.random.randn(k,j) for j,k in zip(size_net[:-1],size_net[1:])]
    
    def train_network_batch(self,x_train,y_train,eta,iteration):
        """
        Batch gradient descent to update the ANN
        
        Parameters
        ----------
        x_train: np.array of input data
        y_train: array-like of labels indicating correct value (scalar)
        eta: stepsize to update the weights and biases with
        iteration: number of iterations to do gradient descent
        """
        for it in range(iteration):
            print('Iteration number: ',it,' out of',iteration)
            print('Evaluate:',self.eval_data(x_train,y_train))
            self.update_weights_back(x_train,y_train,eta)   
            
    def train_network_SGD(self,x_train,y_train,minibatch_size,eta,iteration):
        """
        Mini batch gradient descent to update the ANN
        
        Parameters
        ----------
        x_train: np.array of input data
        y_train: array-like of labels indicating correct value (scalar)
        minibatch: size of each subset to update the ANN
        eta: stepsize to update the weights and biases with
        iteration: number of iterations to do gradient descent
        """
        accuracy = np.zeros(iteration)
        error = np.zeros(iteration)
        for it in range(iteration):
            Training_data = list(zip(x_train.transpose(),y_train))            
            random.shuffle(Training_data)
            mini_batches = [Training_data[k:k+minibatch_size] for k in np.arange(0,len(Training_data),minibatch_size)]
            for train_mini in mini_batches:
                x_mini_train,y_mini_train = zip(*train_mini)
                self.update_weights_back(x_mini_train,y_mini_train,eta)
            accuracy[it],error[it] = self.eval_data(x_train,y_train)
            print('Iteration number: ',it,' out of',iteration)
            print('Accuracy:',accuracy[it])
            print('Error:',error[it])
        return accuracy,error   

    def back_prop(self,x_in,y_in):
        #Back prop for one training image
        Der_weight = [np.zeros(W.shape) for W in self.weight]
        Der_bias = [np.zeros(b.shape) for b in self.bias]
        
        #forward prop to get a and z for each layer
        a_in = x_in
        a_in_net = [a_in] #list to store a in each layer
        z_net = [] #list to store z values in each layer
        for b,W in zip(self.bias,self.weight):
            z = np.matmul(W,a_in).reshape(-1,1)+b
            z_net.append(z) #save  
            a_in = self.sigmoid(z)
            a_in_net.append(a_in) #save  
        
        #backward prop    
        delta_L = np.multiply(self.nabla_cost(a_in_net[-1],y_in),self.sigmoid_prime(z_net[-1]))
          
        Der_weight[-1] = np.matmul(delta_L,a_in_net[-2].transpose())  
        Der_bias[-1] = delta_L
        delta_l = delta_L
         
        for l in range(2,self.num_lay):
            z_l = z_net[-l]
            sig_prim_l = self.sigmoid_prime(z_l)
            delta_l = np.multiply(np.matmul(self.weight[1-l].transpose(),delta_l),sig_prim_l)
            Der_weight[-l] = np.matmul(delta_l,a_in_net[-l-1].reshape(-1,1).transpose())  
            Der_bias[-l] = delta_l
        
        return Der_weight,Der_bias
    
    def update_weights_back(self,x_train,y_train,eta):
        #Gives the averaged weight for the matrix to be adjusted with
        DELTA_weight = [np.zeros(W.shape) for W in self.weight]
        DELTA_bias = [np.zeros(b.shape) for b in self.bias]
        
        for k in range(np.size(y_train)):
            delta_W,delta_b = self.back_prop(x_train[k],y_train[k])
            DELTA_weight = [D_w+dn_w for D_w,dn_w in zip(DELTA_weight,delta_W)]
            DELTA_bias = [D_b+dn_b for D_b,dn_b in zip(DELTA_bias,delta_b)] 
        #update the weights using steepest descent
        self.weights = [w-(eta/len(y_train))*Dw for w,Dw in zip(self.weight,DELTA_weight)]
        self.bias = [b-(eta/len(y_train))*Db for b,Db in zip(self.bias,DELTA_bias)]
                
    #Gradient of cost function        
    def nabla_cost(self,a_L,y_in):
        nabla_L = a_L
        nabla_L[y_in]=nabla_L[y_in]-1
        return nabla_L     
    #cost function 
    def cost(self,a_L,y_in):
        cost_L = a_L
        cost_L[y_in] = cost_L[y_in]-1
        return (np.linalg.norm(cost_L)**2)/2  
                             
    def eval_data(self,test_data,test_labels):
        test_res = [np.argmax(self.forward_prop(test_data[:,k])) for k in range(np.size(test_labels))]
        accuracy = sum(int(test_res[k] == test_labels[k]) for k in range(np.size(test_labels)))/ np.size(test_labels)   
        error_sum = sum(self.cost(self.forward_prop(test_data[:,k]),test_labels[k]) for k in range(np.size(test_labels)))        
        return accuracy,error_sum
    
    def forward_prop(self,a):
        #Gives the last unit
        for b,W in zip(self.bias,self.weight):
            z = np.matmul(W,a).reshape(-1,1)+b 
            a = self.sigmoid(z)
        return a    
    
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
