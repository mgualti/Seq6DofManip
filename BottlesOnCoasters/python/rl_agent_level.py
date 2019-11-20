'''Class and utilities for one level of an HSA agent.'''

# python
import os
import pickle
from copy import copy
from time import time
# scipy
from numpy.linalg import inv, norm
from numpy.random import choice, permutation, rand, randint, shuffle
from numpy import argmax, arange, array, cos, concatenate, dot, empty, exp, eye, flipud, floor, \
  logical_not,isinf, max, maximum, mean, min, minimum, ones, pi, prod, repeat, reshape, round, \
  sign, sin, squeeze, sqrt, stack, sum, unravel_index, where, zeros
# drawing
from matplotlib import pyplot
from skimage.draw import ellipse_perimeter, line
# tensorflow
import tensorflow
from tensorflow import keras
# openrave
import openravepy
# self

# AGENT ============================================================================================

class RlAgentLevel():

  def __init__(self, level, isGrasp, isOrient, params):
    '''TODO'''

    # parameters

    self.level = level
    self.isGrasp = isGrasp
    self.isOrient = isOrient
    self.params = params
    self.tMax = params["tMax"]
    self.trainEvery = params["trainEvery"]
    self.maxExperiences = params["maxExperiences"]
    self.epsilonMin = params["epsilonMin"]
    self.gamma = params["gamma"]
    self.includeTime = params["includeTimeGrasp"] if isGrasp else params["includeTimePlace"]
    self.nGraspOrientations = params["nGraspOrientations"]
    self.nPlaceOrientations = params["nPlaceOrientations"]
    self.nEpochs = params["nEpochs"]
    self.batchSize = params["batchSize"]
    self.imP = params["imP"]
    self.plotImages = params["plotImages"]
    
    # at least half the time steps are grasps
    self.minExperiencesToTrain = self.trainEvery * (self.tMax / 2)
    
    # initialize
    
    self.experiences = []
    self.Q = self.GenerateNetworkModel(isGrasp)
    
    shape = []
    for i in xrange(1, len(self.Q[1].outputs[0].shape)):
      shape.append(int(self.Q[1].outputs[0].shape[i]))
    self.outputShape = tuple(shape)
    self.nActionSamples = prod(self.outputShape)
    
    typeString = "Grasp" if isGrasp else "Place"
    print("{} level {} has {} outputs.".format(typeString, self.level, self.outputShape)) 

  def AddExperience(self, experience):
    '''Adds an experience to the experience database.
    - Input experience: A tuple of (s, a, r, ...). The contents of ... depends on the update rule
      (e.g. Monte Carlo, Q-learning, or Sarsa).
    - Returns None.
    '''
    
    self.experiences.append(experience)

  def EvaluateActions(self, image):
    '''Run forward propagation and get approximated action-values.
    - Input handImage: Image representing hand contents.
    - Input targImage: Image representing action and current sensed volume.
    - Input flags: State flags, (inHandBit, episodeTime).
    - Input actions: List of candidate actions in the agent's encoding.
    - Return values: Values, one for each action, approximating the q-values.
    '''
    
    dummyIn = zeros((1, 1, 2), dtype='int32') if self.isOrient else zeros((1, 1, 4), dtype='int32')
    values = self.Q[1].predict([array([image]), dummyIn])
    return squeeze(values)
    
  def EvaluateActionsMultiple(self, images):
    '''TODO'''
    
    batchSize = len(images)
    dummyIn = zeros((batchSize, 1, 2), dtype='int32') if self.isOrient else \
      zeros((batchSize, 1, 4), dtype='int32')
    values = self.Q.predict([array(images), dummyIn])
    return squeeze(values)

  def GenerateNetworkModel(self, graspNetwork):
    '''Generates tensorflow model for the deep network used to approximate the Q-function at this
    level. Must be called during initialization if a model is not loaded from file.
    - Input params: Dictionary of hyperparameters.
    - Returns None.
    '''
    
    params = self.params
    weightDecay = params["weightDecay"]
    optimizer = params["optimizer"].lower()
    baseLearningRate = params["baseLearningRate"]

    # architecture
    
    in1Shape = (self.imP, self.imP, 1 + (not graspNetwork) + self.includeTime)
    in2Shape = (1, 2) if self.isOrient else (1, 4)
    nDenseOutputs = self.nGraspOrientations if graspNetwork else self.nPlaceOrientations
      
    in1 = keras.Input(shape=in1Shape, dtype=tensorflow.float32)
    in2 = keras.Input(shape=in2Shape, dtype=tensorflow.int32)
    h1 = keras.layers.Conv2D(params["conv1Outputs"], kernel_size=params["conv1KernelSize"], \
      strides=params["conv1Stride"], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(in1)
    h1 = keras.layers.Conv2D(params["conv2Outputs"],  kernel_size=params["conv2KernelSize"], \
      strides=params["conv2Stride"], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    h1 = keras.layers.Conv2D(params["conv3Outputs"], kernel_size=params["conv3KernelSize"], \
      strides=params["conv3Stride"], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    if self.isOrient:
      h1 = keras.layers.Flatten()(h1)
      h1 = keras.layers.Dense(nDenseOutputs, \
        kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    else:
      h1 = keras.layers.Conv2D(params["conv4Outputs"], kernel_size=params["conv4KernelSize"], \
        strides=params["conv4Stride"], padding="same", \
        kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    h2 = keras.layers.Lambda(lambda inputs: tensorflow.gather_nd(inputs[0], inputs[1]))([h1, in2])
    Qtrain = keras.Model(inputs=[in1, in2], outputs=h2)
    Qtest = keras.Model(inputs=[in1, in2], outputs=h1)

    # optimization

    if optimizer == "adam":
      optimizer = keras.optimizers.Adam(baseLearningRate)
    elif optimizer == "rmsprop":
      optimizer = keras.optimizers.RMSprop(baseLearningRate)
    elif optimizer == "sgd":
      optimizer = keras.optimizers.SGD(baseLearningRate)
    else:
      raise Exception("Unsupported optimizer {}.".format(optimizer))

    Qtrain.compile(optimizer=optimizer, loss="MSE")
    
    #typeString = "grasp" if graspNetwork else "place"
    #print("Summary of {} Q-function for level {}:".format(typeString, self.level))
    #Qtrain.summary()
    
    return Qtrain, Qtest

  def GetNumberOfExperiences(self):
    '''Returns the number of entries in the experience replay database currently in memory at this level.'''
    
    return len(self.experiences)

  def LabelDataMonteCarlo(self):
    '''Given a database of (s, g), reorganize into network model inputs and training labels.
    - Returns in1: List of first inputs into network (an image).
    - Returns labels: List of training labels, one for each set of inputs.
    '''
    
    indexShape = (1, 2) if self.isOrient else (1, 4)    
    
    inputs1 = []; inputs2 = []; labels = []
    for d in self.experiences:
      inputs1.append(d[0]) # image
      index = zeros(indexShape, dtype='int32')
      index[0, 1:] = d[1]
      inputs2.append(index) # index
      labels.append(d[2]) # return

    return array(inputs1), array(inputs2), array(labels)

  def LoadExperienceDatabase(self):
    '''Loads the experience database to file.'''

    path = os.getcwd() + "/tensorflow/experiences/experiences_level_" + str(self.level) + ".pickle"
    self.experiences = pickle.load(open(path, "rb"))
    print("Loaded database " + path + ".")

  def LoadQFunction(self):
    '''Loads the network model and weights from the specified file name.'''
    
    directory = os.getcwd() + "/tensorflow/models"
    
    act = "grasp" if self.isGrasp else "place"
    path = directory + "/q_level_" + str(self.level) + "_" + act + ".h5"
    self.Q[0].load_weights(path)
    print("Loaded Q-function " + path + ".")

  def PlotImages(self, o, a, desc):
    '''Produces plots of the robot's observation and selected action.
    - Input o: Image where 1st channel is the target sensed volume and the 2nd channel is the hand
      contents.
    - Input a: Index into the Q-function output which corresponds to the selected action.
    - Input desc: Descriptor corresponding to the current action in the base frame.
    - Returns None.
    '''

    # setup
    It = o[:, :, 0]
    Ih = zeros(It.shape) if self.isGrasp else o[:, :, 1]
    Ir = copy(It); Ig = copy(It); Ib = copy(It)
    
    if self.isOrient:
      
      # determine rotation angle
      R = desc.T[0:3, 0:3]
      axisAngle = openravepy.axisAngleFromRotationMatrix(R)
      angle = norm(axisAngle)
      axis = axisAngle / angle if angle > 0 else array([0.0, 0.0, 1.0])
      angle *= sign(sum(axis))
      
      # draw axis indicator
      c = self.imP / 2
      majorRadius = self.imP / 8
      minorRadius = majorRadius if self.isGrasp else majorRadius / 2
      xx, yy = ellipse_perimeter(c, c, minorRadius, majorRadius, orientation=0)
      Ir[xx, yy] = 1.0
      
      # draw angle indicator
      length = self.imP / 5
      x = -int(length * sin(angle))
      y = int(length * cos(angle))
      xx, yy = line(c, c, c + x, c + y)
      Ir[xx, yy] = 1.0
      xx, yy = line(c, c, c, c + length)
      Ir[xx, yy] = 1.0
    
    else:

      # draw the selection area
      halfWidth = (It.shape[0] * (self.selW / self.imW)) / 2.0
      middle = It.shape[0] / 2.0
      start = int(round(middle - halfWidth))
      end = int(round(middle + halfWidth))
      pixels = arange(start, end + 1)
      if start >= 0 and end < It.shape[0]:
        Ib[start, pixels] = 1.0
        Ib[end, pixels] = 1.0
        Ib[pixels, start] = 1.0
        Ib[pixels, end] = 1.0
  
      # draw robot's selection
      xh = self.actionsInHandFrame[a]
      xi = round(((xh[0:2] * self.imP) / self.imW) + ((self.imP - 1.0) / 2.0)).astype('int32')
      value = (xh[2] + (self.imD / 2.0)) / self.imD
      if xi[0] >= 0 and xi[1] < self.imP and xi[1] >= 0 and xi[1] < self.imP:
        Ir[xi[0], xi[1]] = value
        Ig[xi[0], xi[1]] = 0
        Ib[xi[0], xi[1]] = 0

    # show image
    fig = pyplot.figure()
    Irgb = stack((Ir, Ig, Ib), 2)
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(Irgb, vmin=0.00, vmax=1.00, interpolation="none")
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(Ih, vmin=0.00, vmax=1.00, interpolation="none", cmap="gray")
    fig.suptitle("(Left.) Sensed volume. (Right.) Hand contents.")
    for i in xrange(2):
      fig.axes[i].set_xticks([])
      fig.axes[i].set_yticks([])
    pyplot.show(block=True)

  def PruneDatabase(self):
    '''Removes oldest items in the database until the size is no more than maxEntries.'''
    
    if len(self.experiences) > self.maxExperiences:
      self.experiences = self.experiences[len(self.experiences) - self.maxExperiences:]

  def SaveExperienceDatabase(self):
    '''Saves the experience database to file.'''

    directory = os.getcwd() + "/tensorflow/experiences"
    if not os.path.isdir(directory):
      os.makedirs(directory)
    
    act = "grasp" if self.isGrasp else "place"
    path = directory + "/experiences_level_" + str(self.level) + "_" + act + ".pickle"
    pickle.dump(self.experiences, open(path, "wb"))
    
    print("Saved database " + path + ".")

  def SaveQFunction(self):
    '''Saves the network model and weights to file.'''

    directory = os.getcwd() + "/tensorflow/models"
    if not os.path.isdir(directory):
      os.makedirs(directory)
      
    act = "grasp" if self.isGrasp else "place"
    path = directory + "/q_level_" + str(self.level) + "_" + act + ".h5"
    self.Q[0].save_weights(path)
    print("Saved Q-function " + path + ".")

  def SelectIndexEpsilonGreedy(self, image, unbias):
    '''Selects the next action according to an epsilon-greedy policy.
    - Input handImage: Depth image representing hand contents.
    - Input targImage: Depth image representing next hand pose.
    - Input actions: Action choices, nxm matrix with n actions.
    - Input flags: State flags, (inHand, time).
    - Input epsilon: Number between 0 and 1 indicating the probability of taking a random action.
    - Returns bestIdx: Index into descriptors indicating the action the policy decided on.
    - Returns bestValue: The value estimate of the state-action. Is NaN if the action is selected
      randomly.
    '''
    
    epsilon = 0.0 if unbias else maximum(self.epsilonMin, \
      1.0 - float(len(self.experiences)) / self.maxExperiences)
    
    if rand() < epsilon:
      bestIdx = randint(self.nActionSamples)
      bestIdx = unravel_index(bestIdx, self.outputShape)
      bestValue = float('NaN')
    else:
      values = self.EvaluateActions(image)
      bestIdx = unravel_index(argmax(values), self.outputShape)
      bestValue = values[bestIdx]
      #self.PlotValues(actions, values)
    
    if len(bestIdx) == 1: bestIdx = bestIdx[0]
    return bestIdx, bestValue, epsilon

  def UpdateQFunction(self, inputs1, inputs2, labels):
    '''Trains the neural network model on the input training data and labels.
    - Input inputs:
    - Input labels: Ground truth for the network output.
    - Returns: Average loss averaged over each epoch.
    '''
    
    # print information
    actString = "grasp" if self.isGrasp else "place"
    print("Level={}-{}".format(self.level, actString))
    # decide whether or not there are enough new experiences to train
    if labels.shape[0] < self.minExperiencesToTrain: return 0.0
    # shuffle data
    idxs = arange(labels.shape[0])
    shuffle(idxs)
    inputs1 = inputs1[idxs]
    inputs2 = inputs2[idxs]
    labels = labels[idxs]
    # add batch index
    for i in xrange(inputs2.shape[0]):
      inputs2[i, 0, 0] = i % self.batchSize
    # fit
    history = self.Q[0].fit([inputs1, inputs2], labels, epochs = self.nEpochs, batch_size = \
      self.batchSize, shuffle = False)    
    # compute average loss
    return mean(history.history["loss"])
    
  def UpdateQFunctionMonteCarlo(self):
    '''Trains the Q-function on the current replay database using the Monte Carlo update rule.
    - Returns: Average loss.
    '''

    self.PruneDatabase()
    inputs1, inputs2, labels = self.LabelDataMonteCarlo()

    return self.UpdateQFunction(inputs1, inputs2, labels)