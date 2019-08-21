'''Class and utilities for one level of a HSENS agent.'''

# python
import os
import pickle
from copy import copy
from time import time
# scipy
from matplotlib import pyplot
from numpy.random import choice, permutation, rand, randint, shuffle
from numpy import argmax, arange, array, concatenate, empty, exp, eye, flipud, floor, logical_not, \
  isinf, max, maximum, mean, min, minimum, ones, pi, repeat, reshape, round, squeeze, sqrt, stack, \
  sum, unravel_index, where, zeros
# tensorflow
import tensorflow
from tensorflow import keras
# self
from hand_descriptor import HandDescriptor

# AGENT ============================================================================================

class RlAgentLookaheadLevel():

  def __init__(self, level, params):
    '''Initializes one level of an HSENS agent. This class will be inherited by a specific level.
    - Input level: An integer inticating which level this is, in order of execution, where 0 is the
      first and nLevels-1 is the last.
    - Input params: Hyperparameter dictionary.
    '''

    # parameters

    self.level = level
    self.params = params
    self.tMax = params["tMax"]
    self.maxExperiences = params["maxExperiences"]
    self.epsilonMin = params["epsilonMin"]
    self.gamma = params["gamma"]
    self.actionSpaceSize = params["actionSpaceSize"][level]
    self.modelFolder = params["modelFolder"]
    self.nEpochs = params["nEpochs"]
    self.batchSize = params["batchSize"]
    self.imP = params["imP"]
    
    self.nActionSamples = self.actionSpaceSize[0] * self.actionSpaceSize[1] * self.actionSpaceSize[2]
    
    # initialize
    
    self.experiences = []
    self.Q = []
    
    for i in xrange(2):
      self.experiences.append([])
      self.Q.append(self.GenerateNetworkModel(i == 0))

  def AddExperience(self, experience):
    '''Adds an experience to the experience database.
    - Input experience: A tuple of (s, a, r, ...). The contents of ... depends on the update rule
      (e.g. Monte Carlo, Q-learning, or Sarsa).
    - Returns None.
    '''
    
    idx = 0 if self.IsGraspImage(experience[0]) else 1
    self.experiences[idx].append(experience)

  def EvaluateActions(self, images):
    '''TODO'''
    
    idx = 0 if self.IsGraspImage(images[0]) else 1
    values = self.Q[idx].predict([array(images)])
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
    
    inShape = (self.imP, self.imP, 1) if graspNetwork else (self.imP, self.imP, 2)
      
    inpt = keras.Input(shape=inShape, dtype=tensorflow.float32)
    h = keras.layers.Conv2D(params["conv1Outputs"][self.level], \
      kernel_size=params["conv1KernelSize"][self.level], \
      strides=params["conv1Stride"][self.level], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(inpt)
    h = keras.layers.Conv2D(params["conv2Outputs"][self.level], \
      kernel_size=params["conv2KernelSize"][self.level], \
      strides=params["conv2Stride"][self.level], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h)
    h = keras.layers.Conv2D(params["conv3Outputs"][self.level], \
      kernel_size=params["conv3KernelSize"][self.level], \
      strides=params["conv3Stride"][self.level], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h)
    h = keras.layers.Conv2D(1, kernel_size=params["conv4KernelSize"][self.level], \
      strides=params["conv4Stride"][self.level], padding="same", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h)
    Q = keras.Model(inputs=inpt, outputs=h)

    # optimization

    if optimizer == "adam":
      optimizer = keras.optimizers.Adam(baseLearningRate)
    elif optimizer == "rmsprop":
      optimizer = keras.optimizers.RMSprop(baseLearningRate)
    elif optimizer == "sgd":
      optimizer = keras.optimizers.SGD(baseLearningRate)
    else:
      raise Exception("Unsupported optimizer {}.".format(optimizer))

    Q.compile(optimizer=optimizer, loss="MSE", metrics=[])
    
    typeString = "grasp" if graspNetwork else "place"
    print("Summary of {} Q-function for level {}:".format(typeString, self.level))
    Q.summary()
    
    return Q

  def GetNumberOfExperiences(self):
    '''Returns the number of entries in the experience replay database currently in memory at this level.'''
    
    return len(self.experiences[0]) + len(self.experiences[1])
    
  def IsGraspImage(self, image):
    '''TODO'''
    
    # graspImage: (targ)
    # placeImage: (targ, hand)
    
    return image.shape[2] == 1

  def LabelDataMonteCarlo(self):
    '''Given a database of (s, g), reorganize into network model inputs and training labels.
    - Returns in1: List of first inputs into network (an image).
    - Returns labels: List of training labels, one for each set of inputs.
    '''
    
    inputs = []; labels = []    
    
    for i in xrange(2):
      inpt = []; lbl = []
      for d in self.experiences[i]:
        inpt.append(d[0]) # image
        lbl.append(array([[[d[1]]]])) # return
      inputs.append(array(inpt))
      labels.append(array(lbl))

    return inputs, labels

  def LoadExperienceDatabase(self):
    '''Loads the experience database to file.'''

    path = os.getcwd() + "/tensorflow/" + self.modelFolder + "/experiences_level_" + \
      str(self.level) + ".pickle"
    self.experiences = pickle.load(open(path, "rb"))
    print("Loaded database " + path + ".")

  def LoadQFunction(self):
    '''Loads the network model and weights from the specified file name.'''
    
    directory = os.getcwd() + "/tensorflow/" + self.modelFolder
    
    self.Q = []
    for i in xrange(2):
      act = "grasp" if i == 0 else "place"
      path = directory + "/q_level_" + str(self.level) + "_" + act + ".h5"
      self.Q.append(keras.models.load_model(path))
      print("Loaded Q-function " + path + ".")

  def PlotImages(self, o):
    '''Produces plots of the observation image.'''

    # Setup

    fig = pyplot.figure()
    It = o[:, :, 0]
    Ih = zeros(It.shape) if self.IsGraspImage(o) else o[:, :, 1]

    # Plot target image

    Ir = copy(It)
    Ig = copy(It)
    Ib = copy(It)

    # show image
    Irgb = stack((Ir, Ig, Ib), 2)
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(Irgb, vmin=0.00, vmax=1.00, interpolation="none")
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(Ih, vmin=0.00, vmax=1.00, interpolation="none", cmap="gray")
    fig.suptitle("(Left.) Target sense. (Right.) Hand contents.")
    for i in xrange(2):
      fig.axes[i].set_xticks([])
      fig.axes[i].set_yticks([])
    pyplot.show(block=True)

  def PruneDatabase(self):
    '''Removes oldest items in the database until the size is no more than maxEntries.'''
    
    for i in xrange(2):
      if len(self.experiences[i]) > self.maxExperiences:
        self.experiences[i] = self.experiences[i][len(self.experiences[i]) - self.maxExperiences:]

  def SaveExperienceDatabase(self):
    '''Saves the experience database to file.'''

    directory = os.getcwd() + "/tensorflow/" + self.modelFolder
    if not os.path.isdir(directory):
      os.makedirs(directory)
    
    path = directory + "/experiences_level_" + str(self.level) + ".pickle"
    pickle.dump(self.experiences, open(path, "wb"))
    
    print("Saved database " + path + ".")

  def SaveQFunction(self):
    '''Saves the network model and weights to file.'''

    directory = os.getcwd() + "/tensorflow/" + self.modelFolder
    if not os.path.isdir(directory):
      os.makedirs(directory)
      
    for i in xrange(2):
      act = "grasp" if i == 0 else "place"
      path = directory + "/q_level_" + str(self.level) + "_" + act + ".h5"
      self.Q[i].save(path)
      print("Saved Q-function " + path + ".")

  def SelectIndexEpsilonGreedy(self, descriptors, handImage, unbias, rlEnv):
    '''TODO'''
    
    idx = 0 if handImage.size == 0 else 1
    epsilon = 0.0 if unbias else maximum(self.epsilonMin, \
      1.0 - float(len(self.experiences[idx])) / self.maxExperiences)
    
    if rand() < epsilon:
      bestIdx = randint(self.nActionSamples)
      descriptors[bestIdx].GenerateHeightmap(rlEnv)
      bestValue = float('NaN')
    else:
      # generate images
      images = []
      for d in descriptors:
        image = d.GenerateHeightmap(rlEnv)
        image = concatenate((image, handImage), axis = 2)
        images.append(image)
      # evaluate actions
      values = self.EvaluateActions(images)
      bestIdx = argmax(values)
      bestValue = values[bestIdx]
      #self.PlotValues(actions, values)

    return bestIdx, bestValue, epsilon

  def UpdateQFunction(self, inputs, labels):
    '''Trains the neural network model on the input training data and labels.
    - Input inputs:
    - Input labels: Ground truth for the network output.
    - Returns: Average loss averaged over each epoch.
    '''
    
    totalLoss = 0; totalBatches = 0
    for i in xrange(2):
      if labels[i].shape[0] == 0: continue
      # print information
      actString = "grasp" if i == 0 else "place"
      print("Level={}-{}".format(self.level, actString))      
      # shuffle data
      idxs = arange(labels[i].shape[0])
      shuffle(idxs)
      inputs[i] = inputs[i][idxs]
      labels[i] = labels[i][idxs]
      # fit
      history = self.Q[i].fit(inputs[i], labels[i], \
        epochs=self.nEpochs, batch_size=self.batchSize, shuffle=False)
      # accumulate loss
      nBatches = floor(labels[i].shape[0] / self.batchSize)
      totalLoss += sum(array(history.history["loss"]) * nBatches)
      totalBatches += self.nEpochs * nBatches
    
    # compute average loss
    return totalLoss / totalBatches

  def UpdateQFunctionMonteCarlo(self):
    '''Trains the Q-function on the current replay database using the Monte Carlo update rule.
    - Returns: Average loss.
    '''

    self.PruneDatabase()
    inputs, labels = self.LabelDataMonteCarlo()

    return self.UpdateQFunction(inputs, labels)