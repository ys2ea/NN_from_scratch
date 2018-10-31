#########
## do a simple example. 
## Train a model to learn a function: y = x1^2 - x2^2 + x3*x1 + x3*x2
#########

import Model
import numpy as np

def generate_dataset(N):
	X_train = np.random.normal(0, 0.5, (N, 3))
	Y_train = X_train[:,0]*X_train[:,0] - X_train[:,1]*X_train[:,1] + X_train[:,2]*X_train[:,0] + X_train[:,2]*X_train[:,1]
	#Y_train = X_train[:,0]
	Y_train = Y_train.reshape(-1,1)
	#print("Y: ", Y_train.shape)
	
	return [X_train, Y_train]


#########
Batch_size = 20
N_hidden = 20
lr = 0.0005
N_batch = 200000

model = Model.Neural_network(3, 1, N_hidden, lr)
for i in range(N_batch):
    [batch_X, batch_Y] = generate_dataset(Batch_size)
    terror = model.train(batch_X, batch_Y)
    
    if i%1000 == 0:
        [test_X, test_Y] = generate_dataset(Batch_size)
        predict = model.evaluate(test_X)
        loss = np.mean(np.square(test_Y - predict))
        print("Error at step {}:  {} and {}".format(i, loss, terror))
        
