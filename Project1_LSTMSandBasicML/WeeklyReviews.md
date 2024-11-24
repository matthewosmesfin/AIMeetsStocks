# Weekly Dialogues
A means to answer the burning questions from the prior weeks meeting.

## Answering Questions (11/23/2024) -- Week 1:

1. Why do we need to do data normalization, min_max used in code?
- Data normilizstion is the process of rescaling data and making is more noramlized and thus easier for models to process.
- A for of data normalization is min_max scaling, where the features of data are scaled from 0 to 1 based on the minimum and maximum points. ML aglos perform better when featues are on the same scale.

2. What are the different data normalization or pre-processing techniques that can be applied?
- MinMax Scaler, z score normalization (standard scaler), L2 normilzation

3. What are hyper-parameters? How can we fine tune them?
- To explain what hyper-parameters we must understand what parameters are and how they differ from hyperparameters.
    - Parameters are things that are learned from the data <i>during</i> training
        - think weights and biases in a NN, or the coefficients in a linear regression model
    - Hyperparameters are set by the engineer/scientist <i>prior<i> to training
        - think # of hidden layers in a NN
- We can tune them by testing our model while we adjust the hyperparameters and seeing which leads to a stronger model.

4. What is our training and testing data split? Does this have an impact on model performance?
- Test to train ratio is 192:1089. A higher training to testing split will give us ample amount of data to build our model but we won't have enough to evaluate our performance. On the other hand, a lower training to testing split will give us more to evulate our model but less data to build it.

5. What are the parameters used in the model, price, volume, …?
- (??) confused as to what this is asking