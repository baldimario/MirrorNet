import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import os


class MirrorNetData:
    _layers = []
    _classifier = []
    
    def __init__(self, layers, classifier):
        self._layers = layers
        self._classifier = classifier
    
    def get_layers(self):
        return self._layers
    
    def get_classifier(self):
        return self._classifier
    
class MirrorNet:
    _instance = None
    _layers = []
    _classifier = None
    _epsilon = 1e-5
    _plot = False
    _plot_components_distribution_at_last_layer = False
    _use_bias = True
    _use_relu = True
    _verbose = True
    
    def __init__(self, layers, classifier):
        self._layers = layers
        self._classifier = classifier
    
    @classmethod
    def save(self, name):
        with open(name, 'wb') as output:
            n = MirrorNetData(self._layers, self._classifier)
            pickle.dump(n, output)
    
    @classmethod
    def load(self, name):
        if not os.path.isfile(name):
            return False
            
        with open(name, 'rb') as input:
            n = pickle.load(input)
            self._layers = n.get_layers()
            self._classifier = n.get_classifier()
            return True
            
    @classmethod
    def get_layers(self):
        return self._layers
        
    @classmethod
    def get_classifier(self):
        return self._classifier
        
    @classmethod
    def relu(self, x):
        #epsilon guarantees to discard values that could be affected 
        #by errors due to machine precision
        if self._use_relu:
            return x * (x > self._epsilon)
        else:
            return x
        
    @classmethod
    def build_extractor(self, data, layers=1, input_dim=None, percentage=1, mode='pca'):
        self._layers = []        
        self._input_dim = input_dim if input_dim else data.shape[-1]
        x = data

        for i in range(layers):
            if self._verbose:
                print("Building layer #"+str(i))

            C = None
            eigenvalues = None
            eigenvectors = None
            if mode == 'pca':
                R = np.dot(x.T, x) / x.shape[0]
                eigenvalues, eigenvectors = np.linalg.eig(R)
                #the eigenvalues are yet ordered decrescently

                if percentage > 0:
                    e = np.trace(R)
                    er = e*percentage

                    diagonal = np.diagonal(R)

                    partial_sum = 0
                    index = 0
                    for j in range(diagonal.shape[0]):
                        partial_sum += diagonal[j]
                        if partial_sum >= e-er:
                            index = j
                            break

                    eigenvalues = eigenvalues[0:j]
                    eigenvectors = eigenvectors[0:j]

                #only half of the weights are calculated because the other
                #half it's mirrored during inference to reduce memory consumption
                C = eigenvectors.T

            elif mode == 'hadamard':
                d = 2**(i+1)
                C = hadamard(d).astype('float32')

            elif mode == 'gauss':
                d = 2**(i+1)
                C = np.random.normal(loc=0, scale=1, size=[d, d])

            else:
                print('Invalid Mode {}: pca hadamard gauss'.format(mode))

            y = np.dot(x, C)
            
            #also the bias retain the mirroring property 
            #being the average of the mirrored activations
            
            b = -np.mean(y, axis=0)
            
            if not self._use_bias:
                b = np.zeros(b.shape)
            
            y = y + b

            y = np.concatenate((y, -y), axis=-1)
            
            x = self.relu(y)
            
            self._layers.append({
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'weights': C,
                'biases': b
            })
        
        return self

    #build_classifier compute the optimal coefficient to minimize the
    #mean square error (Haykin 1986)
    @classmethod
    def build_classifier(self, data, labels, output_dim=None):
        if self._verbose:
            print("Building classifier")
            
        self.classifier_ = []
        self._output_dim = output_dim if output_dim else labels.shape[-1]
        
        x = self.forward(data)
        x = x - np.mean(x, axis=0) #removing the mean from components
        
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=-1)
        R = np.dot(x.T, x) 
        D = np.dot(x.T, labels)

        M = np.linalg.pinv(R)
        W = np.dot(M, D)
        
        y = np.dot(x, W)
        
        b = -np.mean(y, axis=0)
        
        if not self._use_bias:
            b = np.zeros(b.shape)
        
        y = y + b
        
        self._classifier = {
            'weights': W,
            'biases': b
        }
            
        return self

    @classmethod
    def forward(self, data, stop_at=None):
        x = data

        for i in range(len(self._layers)):
            if self._verbose:
                print("Compute activation layer #"+str(i))
                
            #x = x - np.mean(x, axis=0)
            
            ##########################
            if self._plot:
                R = np.dot(x.T, x) / x.shape[0]
                eigenvalues, eigenvectors = np.linalg.eig(R)
                
                plt.title('prima della relu livello '+str(i))
                plt.plot(np.log(eigenvalues))
                plt.show()
            
            '''
            e = np.trace(R)
            print('Energy['+str(i)+']: '+str(e))
            er = e*0.1
            print('Energy['+str(i)+']*0.1: '+str(er))
            
            diagonal = np.diagonal(R)
            
            partial_sum = 0
            index = 0
            for j in range(diagonal.shape[0]):
                partial_sum += diagonal[j]
                if partial_sum > e-er:
                    index = j
                    break
                    
            print(str(index)+'/'+str(diagonal.shape[0]))
            '''
            ###########################
            
            W = self._layers[i]['weights']
            b = self._layers[i]['biases']

            y = np.dot(x, W) + b

            y = np.concatenate((y, -y), axis=-1)

            x = self.relu(y)

            if stop_at == i:
                return x
            
            if self._plot_components_distribution_at_last_layer:
                if i == len(self._layers)-1:
                    for k in range(y.shape[1]):
                        plt.hist(y[:, k])
                        plt.show()
            
            
        return x
        
    #the classify mehtod can be feeded with the already computed data 
    #from network's layers or the data from the input space
    @classmethod
    def classify(self, data, feed=False):
        if self._verbose:
            print("Compute classification")
            
        if feed == False:
            x = self.forward(data)
        else:
            x = data
        
        x = x - np.mean(x, axis=0)
        
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=-1)

        W = self._classifier['weights']
        b = self._classifier['biases']

        y = np.dot(x, W) + b
        y = y - np.mean(y, axis=0)
        
        return y
    
    @classmethod
    def summary(self):
        print('Layers:')
        for i in range(len(self._layers)):
            dim = self._layers[i]['weights'].shape[0]
            print("\tLayer #"+str(i+1)+': '+str(dim)+'x'+str(dim*2))
            
        print('Classifier:')
        dim = self._classifier['weights'].shape[0]
        classes = self._classifier['weights'].shape[1]
        print("\tCoefficients: "+str(dim-1)+"x"+str(classes))
        
    @classmethod
    def inverse(self, y):
        for i in reversed(range(len(self._layers))):
            if self._verbose:
                print("Inverting layer #"+str(i))
                
            if self._use_bias:
                y = y - b
                
            #y = self.relu(y)
            W = self._layers[i]['weights']
            b = np.concatenate((self._layers[i]['biases'], -self._layers[i]['biases']), axis=-1)
            
            
            W = np.concatenate((W, -W), axis=1)
            
            #y = np.dot(y, W.T)
            y = np.dot(y, W.T)
            
            #y = self.relu(y)
            print(y.shape)            
        
        return y
        

        
        
        
        
        
        
        
