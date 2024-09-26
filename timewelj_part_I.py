import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# required inclusion
np.random.seed(42)

# import data
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

# drop columns that will not be used
by_year = (data[data['Year']==2018]).drop(columns=["Continent","Population (historical estimates)","Code"])

# remove missing values from columns 
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

# create np.array for gdp and happiness where happiness score is above 4.5
happiness=[]
gdp=[]
for row in df.iterrows():
    if row[1]['Cantril ladder score']>4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

class linear_regression():
 
    def __init__(self,x_:list,y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self,):

        # normalize the values
        xMean = np.mean(self.input)
        xStd = np.std(self.input)
        xNormalized = (self.input - xMean)/xStd

        # i made the design decision to not split my training set, to minimize my training MSE

        # arrange in matrix format
        x = np.column_stack((np.ones(len(xNormalized)),xNormalized))

        # normalize the values
        yMean = np.mean(self.target)
        yStd = np.std(self.target)
        yNormalized = (self.target - yMean)/yStd

        # arrange in matrix format
        y = (np.column_stack(yNormalized)).T

        # return processed data
        return x,y

    def trainGD(self, X, Y, iterationCount, learningRate):

        # number of samples
        n = len(X)

        # compute and return beta
        beta = np.random.randn(2,1)
        for i in range(1,iterationCount):
            gradients = 2/n * (X.T).dot(X.dot(beta) - Y)
            beta = beta - learningRate * gradients
        return beta
    
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
    
# beta object for saving data relating to beta values
class beta():
    def __init__(self,values:list,learningRate:int,iterationCount:float, yPredicted:list, color:str):
        self.values = values
        self.learningRate = learningRate
        self.iterationCount = iterationCount
        self.yPredicted = yPredicted
        self.color = color

#instantiate the linear_regression class  
lr = linear_regression(gdp,happiness)

# preprocess the inputs
x,y = lr.preprocess()

# access the 1st column (the 0th column is all 1's)
xRavel = x[...,1].ravel()

# set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(xRavel,y)

# color values and beta array
colors = ["red","blue","yellow","orange","green"]
betas = []

# create and display 5 different betas
for i in range(1,6):

    # experiment with different learning rates and iteration counts
    iterationCount = 3**i
    learningRate = 0.12 + 0.02*i*(-1)**i

    # compute beta
    betaValues = lr.trainGD(x,y,iterationCount,learningRate)

    # use the computed beta for prediction
    yPredicted = lr.predict(x,betaValues)

    # display the line predicted by beta
    ax.plot(xRavel,yPredicted,color = colors[i-1],alpha = 0.5)

    # save for cross validation
    betas.append(beta(betaValues, learningRate, iterationCount, yPredicted, colors[i-1]))

    # print all beta values along with their corresponding epochs and learning rates
    print("Line #" + str(i) + ": B0 = " + str(betaValues[0]) + " B1 = " + str(betaValues[1]) + " Learning Rate = " + str(learningRate) + " Epoch = " + str(iterationCount) + " Color = " + colors[i-1])

# set the x-labels
ax.set_xlabel("GDP per capita")

# set the y-labels
ax.set_ylabel("Happiness")

# set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

# show the plot (graph 1)
plt.show()

# the following code is for graph 2

# use cross validation to choose the best beta, default to the first
yPredicted = (np.column_stack(betas[0].yPredicted)).T
minMSE = lr.MSE(y, yPredicted)
bestBeta = betas[0]

for currentBeta in betas:

    # arrange in matrix format
    yPredicted = (np.column_stack(currentBeta.yPredicted)).T

    # compute MSE for each beta
    currentMSE = lr.MSE(y, yPredicted)
    print("currentMSE: " + str(currentMSE))

    # the best beta has minimum MSE
    if currentMSE < minMSE:
        minMSE = currentMSE
        bestBeta = currentBeta

print("minMSE: " + str(minMSE))

# compute OLS beta
betaOLS = lr.trainOLS(x,y)

# use the computed beta for OLS prediction
yPredictedOLS = lr.predict(x,betaOLS)

# set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(xRavel,y)

# display the line predicted by OLS beta and X
ax.plot(xRavel,yPredictedOLS,color="black",alpha=0.5)

# print OLS beta value
print("Learned Through OLS:     B0 = " + str(betaOLS[0]) + " B1 = " + str(betaOLS[1]) + " Color = gray")

# display the line predicted by GD beta and X
ax.plot(xRavel,bestBeta.yPredicted,color=bestBeta.color,alpha=0.5)

# print GD beta value along with epoch and learning rate
print("Best Learned Through GD: B0 = " + str(bestBeta.values[0]) + " B1 = " + str(bestBeta.values[1]) + " Learning Rate = " + str(bestBeta.learningRate) + " Epoch = " + str(bestBeta.iterationCount) + " Color = " + bestBeta.color)

# set the x-labels
ax.set_xlabel("GDP per capita")

# set the y-labels
ax.set_ylabel("Happiness")

# set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

# show the plot (graph 2)
plt.show()