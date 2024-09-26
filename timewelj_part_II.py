import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# required inclusion
np.random.seed(42)

# import data
data = pd.read_csv("datasets/training_data.csv")

# drop columns that will not be used
corr_matrix = data.iloc[: , 1:].corr()
rings = data.pop(data.columns[-1])
features = data.iloc[: , 1:]

# visualize all features and their relationship to the age of the abalone
print("All features and their correlation to Rings: ")
print(corr_matrix["Rings"].sort_values(ascending=False))
fig = plt.figure(figsize=(20, 10))
i=1
for column in features:
    # create subplot
    plt.subplot(1,7,i)
    # display the X and Y points
    plt.scatter(features[column],rings)
    # set the x-labels
    plt.xlabel(column)
    # set the y-labels
    plt.ylabel("Rings")
    # set the title
    plt.title(column + " vs Rings")
    i+=1
# show the plot
fig.tight_layout()
plt.show()

class linear_regression():
 
    def __init__(self,x_,y_) -> None:
        self.input = []
        for column in x_:
            self.input.append(np.array(x_[column]))
        self.target = np.array(y_)

    def preprocess(self,):
        # prepare matrices (two sets for the 2-fold cross validation)
        xTrainOne = np.ones(len(self.input[0][len(self.input[0])//5:]))
        xValidateOne = np.ones(len(self.input[0][:len(self.input[0])//5]))
        xTrainTwo = np.ones(len(self.input[0][:len(self.input[0])-len(self.input[0])//5]))
        xValidateTwo = np.ones(len(self.input[0][len(self.input[0])-len(self.input[0])//5:]))

        for x in self.input:
            # normalize the values
            xMean = np.mean(x)
            xStd = np.std(x)
            xNormalized = (x - xMean)/xStd

            # split the values (training set and validation set for cross validation 80/20 split)
            currentXTrainOne = xNormalized[len(xNormalized)//5:]
            currentXValidateOne = xNormalized[:len(xNormalized)//5]
            currentXTrainTwo = xNormalized[:len(xNormalized)-len(xNormalized)//5]
            currentXValidateTwo = xNormalized[len(xNormalized)-len(xNormalized)//5:]

            # arrange in matrix format
            xTrainOne = np.column_stack((xTrainOne,currentXTrainOne))
            xValidateOne = np.column_stack((xValidateOne,currentXValidateOne))
            xTrainTwo = np.column_stack((xTrainTwo,currentXTrainTwo))
            xValidateTwo = np.column_stack((xValidateTwo,currentXValidateTwo))

        # normalize the values
        yMean = np.mean(self.target)
        yStd = np.std(self.target)
        yNormalized = (self.target - yMean)/yStd

        # split the values
        yTrainOne = yNormalized[len(xNormalized)//5:]
        yValidateOne = yNormalized[:len(yNormalized)//5]
        yTrainTwo = yNormalized[:len(yNormalized)-len(yNormalized)//5]
        yValidateTwo = yNormalized[len(yNormalized)-len(yNormalized)//5:]

        # arrange in matrix format
        yTrainOne = (np.column_stack(yTrainOne)).T
        yValidateOne = (np.column_stack(yValidateOne)).T
        yTrainTwo = (np.column_stack(yTrainTwo)).T
        yValidateTwo = (np.column_stack(yValidateTwo)).T

        # return processed data
        return xTrainOne, xValidateOne, yTrainOne, yValidateOne, xTrainTwo, xValidateTwo, yTrainTwo, yValidateTwo
    
    def trainOLS(self, X, Y):
        # compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X, beta):
        # predict using beta
        Y_hat = X*beta.T
        return np.sum(Y_hat,axis=1)
    
    def MSE(self, real_data, predicted_data):
        # compute difference of real and predicted data
        n = len(real_data)
        return 1/n * sum((real_data - predicted_data)**2)

# instantiate the linear_regression class  
lr = linear_regression(features,rings)

# preprocess the inputs
xTrain,xValidate,yTrain,yValidate,xTrainTwo,xValidateTwo,yTrainTwo,yValidateTwo = lr.preprocess()

# compute OLS beta using training set
beta = lr.trainOLS(xTrain,yTrain)

# use the computed beta for prediction on validation set
yPredicted = lr.predict(xValidate,beta)

# evaluate the performance of model using cross validation (2-fold validation)
betaTwo = lr.trainOLS(xTrainTwo,yTrainTwo)
yPredictedTwo = lr.predict(xValidateTwo,beta)

# arrange in matrix format and compute MSE
MSE = lr.MSE(yValidate, (np.column_stack(yPredicted)).T)
print("MSE One: " + str(MSE))
MSETwo = lr.MSE(yValidateTwo, (np.column_stack(yPredictedTwo)).T)
print("MSE Two: " + str(MSETwo))

# choose the best beta value
if MSETwo < MSE:
    MSE = MSETwo
    beta = betaTwo
    yPredicted = yPredictedTwo
    xValidate = xValidateTwo
    yValidate = yValidateTwo

# report your B′ values
msg = "Beta Values:"
i = 0
for b in beta:
    msg += " B" + str(i) + " = " + str(b)
    i+=1
print(msg)

# visualize your fit
fig = plt.figure(figsize=(20, 10))
i=1
for column in features:
    # access the ith column (the 0th column is all 1's)
    xValidateRavel = xValidate[...,i].ravel()
    # create subplot
    plt.subplot(1,7,i)
    # display the X and Y points
    plt.scatter(xValidateRavel,yValidate,label="real")
    # display the line predicted by OLS beta and X
    plt.scatter(xValidateRavel,yPredicted, color="black",alpha=0.4,label="predicted")
    # set the legend
    plt.legend(loc="upper left")
    # set the x-labels
    plt.xlabel(column)
    # set the y-labels
    plt.ylabel("Rings")
    # set the title
    plt.title(column + " vs Rings")
    i+=1
# show the plot
fig.tight_layout()
plt.show()