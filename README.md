In this folder I collect projects where I explore the mathematics behind various Machine learning algorithms by building them myself. 

## Contains
### ANN: 
A artificial neural network implemented from scratch. I learn the weights and biases through batch stochastic gradient descent. I test the ANN by identifying handwritten numbers (a classic data set found here [Number data set](http://yann.lecun.com/exdb/mnist/)). 

The backward propagation, 
<p align="center">
  <img width="200" src="https://user-images.githubusercontent.com/32745301/34848820-dd8ea690-f6e5-11e7-88ce-a9e736082179.png">
</p>

### RL:
Reinforcement learning implented from scratch. I investigate the use of combined modules having their own Q-table. I apply Gibbs policy improvement. I apply this to an environment containing various rewards.  

Gibbs policy improvement, 
<p align="center">
  <img width="224" alt="screen shot 2018-01-11 at 3 55 36 pm" src="https://user-images.githubusercontent.com/32745301/34849549-5488120c-f6e8-11e7-9791-c94ee13de4d1.png">
</p>

Q-table update, 
<p align="center">
  <img width="423" alt="screen shot 2018-01-11 at 3 55 25 pm" src="https://user-images.githubusercontent.com/32745301/34849548-547c54b2-f6e8-11e7-8aa8-c8372e8fb186.png">
</p>


### Gaussian Process
Gaussian process used to predict time-series data for motion movement. The hyper parameters are found by maximizing the log likelyhood function
<p align="center">
<img width="320" alt="screen shot 2018-02-22 at 4 18 33 pm" src="https://user-images.githubusercontent.com/32745301/36567668-cdd69f8c-17ec-11e8-93b5-42d49cc0e3ad.png">
</p>

where Q is the kernel given by, 
<p align="center">
<img width="444" alt="screen shot 2018-02-22 at 4 19 04 pm" src="https://user-images.githubusercontent.com/32745301/36567748-24e930b4-17ed-11e8-99d7-ec0e8a63c70c.png"> 
</p>

I use steepest ascent to find the hyper parameters, 
<p align="center">
<img width="196" alt="screen shot 2018-02-22 at 4 19 11 pm" src="https://user-images.githubusercontent.com/32745301/36567813-59e12f6a-17ed-11e8-91c2-2e6c600abcce.png">
</p>



## Resources:
[Excelent list of free, open source books on machine learning, statistics, data-mining, etc.](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md)
