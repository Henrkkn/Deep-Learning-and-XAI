
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as keras
import matplotlib.pyplot as plt



# -----------------------------------------------------------------------------------------------------------------------
########## 1. Linear model with simple data ########## 
# -----------------------------------------------------------------------------------------------------------------------

X = tf.constant([[1,0], [1,2]], tf.float32)
Y = tf.constant([[2], [4]], tf.float32)

# Beta = (X'X)^(1-)X'y
# Matric multiply X by X's transpose and invert
XX1 = tf.linalg.inv(tf.matmul(tf.transpose(X), X))

# Matrix multply X'Y
XY = tf.matmul(tf.transpose(X), Y)

# Matrix multiply beta_1 by Y
beta = tf.matmul(XX1, XY)

# Print coefficient vector
# Print results
print("Matrix X:")
print(X.numpy())

print("\nVector Y:")
print(Y.numpy())

print("\nCoefficient Vector (Beta):")
print(beta.numpy())


# OLS by hand with function
@tf.autograph.experimental.do_not_convert
def ols_predict(X, beta):
    y_hat = tf.matmul(X, beta)
    return y_hat

# Predict Y using X and beta
predict = ols_predict(X, beta)
print(predict.numpy())
print(X.numpy())

# Next we sole the model iteratively with a stochastic gradient descent algorithm and use MSE as a loss function
# Define the sequential model
ols = tf.keras.Sequential()

# Add dense layer with linear activation.
ols.add(tf.keras.layers.Dense(1, input_shape = (2,),
use_bias = False , activation = 'linear'))

# Set optimizer and loss function
ols.compile(optimizer = "SGD", loss = "mse")

# Train the model, where we hide the each epoch
ols.fit(X, Y, epochs = 100, verbose = 0)

# Print out the parameter estimates
print(ols.weights[0].numpy())



# -----------------------------------------------------------------------------------------------------------------------
########## 2. OLS and LAD (Least Absolute Deivations) ##########
# -----------------------------------------------------------------------------------------------------------------------
# Set the number of observations (N) and samples (S)
S = 1000
N = 10000

# Set the true values of parameters
alpha = tf.constant([1.], tf.float32)
beta = tf.constant([3.], tf.float32)

# Draw the independent variable and error
X = tf.random.normal([N, S])
epsilon = tf.random.normal([N, S], stddev = 0.25)

# Compute dependent variable
Y = alpha + beta*X + epsilon

# Estimate beta via closed form solution for OLS
XX1 = tf.linalg.inv(tf.matmul(tf.transpose(X), X))  # X'X^-1

# Matrix multiply X'Y
XY = tf.matmul(tf.transpose(X), Y)

# Matrix mutliply beta_1 with Y
betaEst = tf.matmul(XX1, XY)
diags = (tf.linalg.tensor_diag_part(betaEst).numpy())  # Extract the diagonal elements, estimated beta is along the diagonal

# Print out the result
print(diags)
print(beta.numpy())
print(np.mean(diags))
print(np.std(diags))

"""Next we use the LAD and the MSE for the LAD, which is to minimize the mean absolute errors and not the square"""
# Draw initial values randomly
alphaHat0 = tf.random.normal([1], stddev=5.0)
betaHat0 = tf.random.normal([1], stddev=5.0)

# Define function to compute MAE loss
def maeLoss (alphaHat, betaHat, xSample, ySample):
    prediction = alphaHat + betaHat*xSample
    error = ySample - prediction
    absError = tf.abs(error)
    return tf.reduce_mean(absError)

""""We use the stochastic gradient descent (SGD) optimizer to minimize the loss function, we have a parameter-minimizer w, learning rate
 and the gradient.
 In tensorflow we do this using optimizers.SGD()
 """
# Learning rate defines how fast the algorithm moves
my_opt = tf.optimizers.SGD(learning_rate = 0.1)

# Define variables
alphaHat = tf.Variable(alphaHat0, tf.float32)
betaHat = tf.Variable(betaHat0, tf.float32)

# Initialize empty list to store the results
alphaHist, betaHist, histLoss = [], [], []
print(alphaHat)
# Perform minimization and retain parameter updates
for j in range(500):
    with tf.GradientTape() as tape:
        loss = maeLoss(alphaHat, betaHat, X[:,0], Y[:, 0])
    histLoss.append(loss.numpy())
    gradients = tape.gradient(loss, [alphaHat, betaHat])
    my_opt.apply_gradients(zip(gradients, [alphaHat, betaHat]))
    alphaHist.append(alphaHat.numpy()[0])
    betaHist.append(betaHat.numpy()[0])


print(np.shape(alphaHist))
print((alphaHist[1:10]))

"""We obtain 'many' solutions for the optimisation problem, which we can evaluate using graphs:"""
params = pd.DataFrame(np.transpose(np.vstack([alphaHist, betaHist])), columns = ["alphaHat", "betaHat"])

# Plot the parameter estimates over iterations
params.plot(figsize=(10, 7))
plt.title("Parameter Estimates over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Parameter Estimates")
plt.show()  # Show the plot

# Plot histograms of the estimates
params.hist(figsize=(10, 7))
plt.suptitle("Histograms of Parameter Estimates")
plt.show()  # Show the histograms



# -----------------------------------------------------------------------------------------------------------------------
# 3. Partially Linear Models
# Often we want to work with non linear models, for those models no closes form solution often exists
# -----------------------------------------------------------------------------------------------------------------------

# Set the true values of parameters
alpha = tf.constant([1.], tf.float32)
beta = tf.constant([3.], tf.float32)
theta = tf.constant([0.05], tf.float32)
epsilon = epsilon

# Draw the independent and error
X = tf.random.normal([N, S])
Z = tf.random.normal([N, S])

# The dependent variable
Y = alpha + beta*X + tf.exp(theta*Z) + epsilon

# Draw initial values randomly.
alphaHat0 = tf.random.normal([1], stddev=5.0)
betaHat0 = tf.random.normal([1], stddev=5.0)
thetaHat0 = tf.random.normal([1], mean = 0.05,                 
            stddev=0.10)

# Compute prediction.
def plm(alphaHat, betaHat, thetaHat, xS, zS):
	prediction = alphaHat + betaHat*xS + tf.exp(thetaHat*zS)
	return prediction


# Define the function to compute MAE
def maeloss(alphaHat, betaHat, thetaHat, xS, zS, yS):
     yHat = plm(alphaHat, betaHat, thetaHat, xS, zS )
     return tf.losses.mae(yS, yHat)


# ------------------------------------------------------------
# Train a partially linear regression model.
# ------------------------------------------------------------

# Define variables.
alphaHat = tf.Variable(alphaHat0, tf.float32)
betaHat = tf.Variable(betaHat0, tf.float32)
thetaHat = tf.Variable(thetaHat0, tf.float32)

# Instantiate optimizer.
opt = tf.optimizers.SGD()

alphaHist, betaHist, thetaHist = [], [], []

# Perform optimization.
for i in range(5000):
    with tf.GradientTape() as tape:
        loss = maeLoss(alphaHat,betaHat,thetaHat,X[:,0],Z[:,0], Y[:,0])   
    gradients = tape.gradient(loss,[alphaHat, betaHat,thetaHat])
    opt.apply_gradients(zip(gradients, [alphaHat, betaHat,thetaHat]))
    # Update list of parameters.
    alphaHist.append(alphaHat.numpy()[0])
    betaHist.append(betaHat.numpy()[0])
    thetaHist.append(thetaHat.numpy()[0])	

# Plot outputs
params = pd.DataFrame(np.transpose(np.vstack([alphaHist,betaHist,thetaHist])), columns = ['alphaHat', 'betaHat','thetaHat'])
plt = params.plot()
params.hist()

print(alphaHat.numpy())
print(betaHat.numpy())
print(thetaHat.numpy())



# -----------------------------------------------------------------------------------------------------------------------
# 4. Non Linear Regression Model with real data
# The exchange rate is modelled as a random walk with a threshold autoregressive model
# -----------------------------------------------------------------------------------------------------------------------

# We first define our data path to the CSV
data_path = r"C:\Users\henrik.knudsen\OneDrive - BI Norwegian Business School (BIEDU)\Desktop\BI-Skole\3 Semester\DL and XAI\Labs\DEXUSUK.csv"

# Load the data using pandas
data = pd.read_csv(data_path)
print(data.head(2))

# Convert log echange rate to numpy array
e_str = np.array(data["DEXUSUK"])
print(e_str.dtype)

# We need to define the data as a float, there are also some missing values
e_level = np.delete(e_str, np.where(e_str == '.'), axis = 0)
e_level = np.array(e_level, float)

# We take the log of the values
e = np.log(e_level)

# Identify the exchange decrease greater than 2%
de = tf.cast(np.diff(e[:-1]) < -0.02, tf.float32)

# Define the lagged exchange rate as a constant
le = tf.constant(e[1:-1], tf.float32)

# Define the exhange rate as a constant
e = tf.constant(e[2:], tf.float32)


# Define the model
def tar(rho0Hat, rho1Hat, le, de):
     # Compute the regime-specific prediction
     regime0 = rho0Hat*le
     regime1 = rho1Hat*le

     # Compute the prediction for regime
     prediction = regime0*de + regime1*(1-de)
     return prediction


# Define loss.
def maeLoss(rho0Hat, rho1Hat, e, le, de):
	ehat = tar(rho0Hat, rho1Hat, le, de)
	return tf.losses.mae(e, ehat)


# Define variables.
rho0Hat = tf.Variable(0.80, tf.float32)
rho1Hat = tf.Variable(0.80, tf.float32)

# Define optimizer.
opt = tf.optimizers.SGD()

rho0Hist , rho1Hist = [],[] 

# Perform minimization.
for i in range(200):
    with tf.GradientTape() as tape:
        loss = maeLoss(rho0Hat, rho1Hat, e, le, de)   
    gradients = tape.gradient(loss,[rho0Hat, rho1Hat])
    opt.apply_gradients(zip(gradients, [rho0Hat, rho1Hat]))
    # Update list of parameters.
    rho0Hist.append(rho0Hat.numpy())
    rho1Hist.append(rho1Hat.numpy())
 
    

# Plot outputs
params = pd.DataFrame(np.transpose(np.vstack([rho0Hist,rho1Hist])), columns = ['rho0Hat', 'rho1Hat'])

# Plotting the line plot
params.plot(figsize=(10, 5))
plt.title('Line Plot of rho0Hat and rho1Hat')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# Plotting the histograms
axes = params.hist(figsize=(10, 5), bins=20)
plt.suptitle('Histograms of rho0Hat and rho1Hat')  # Add a title to the entire figure

# Adjust layout to prevent overlap between the suptitle and the plots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the histograms
plt.show()