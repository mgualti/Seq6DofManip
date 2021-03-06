'''Class and utilities for one level of an HSA agent.'''

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

# AGENT ============================================================================================

class RlAgentLevel():

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
    self.modelFolder = params["modelFolder"]
    self.nEpochs = params["nEpochs"]
    self.batchSize = params["batchSize"]
    
    self.imP = params["imP"]
    self.imD = params["imD"][level]
    self.imW = params["imW"][level]
    self.imDNext = params["imD"][level + 1]
    self.imWNext = params["imW"][level + 1]
    self.selD = params["selD"][level]
    self.selW = params["selW"][level]
    
    # initialize
    
    self.experiences = []
    self.Q = []
    
    for i in xrange(2):
      self.experiences.append([])
      self.Q.append(self.GenerateNetworkModel(i == 0))
    
    self.outputShape = (int(self.Q[0][1].outputs[0].shape[1]), \
                        int(self.Q[0][1].outputs[0].shape[2]), \
                        int(self.Q[0][1].outputs[0].shape[3]))
    print("Level {} has {} outputs.".format(self.level, self.outputShape))
    self.nActionSamples = self.outputShape[0] * self.outputShape[1] * self.outputShape[2]

  def AddExperience(self, experience):
    '''Adds an experience to the experience database.
    - Input experience: A tuple of (s, a, r, ...). The contents of ... depends on the update rule
      (e.g. Monte Carlo, Q-learning, or Sarsa).
    - Returns None.
    '''
    
    idx = 0 if self.IsGraspImage(experience[0]) else 1
    self.experiences[idx].append(experience)

  def EvaluateActions(self, image):
    '''Run forward propagation and get approximated action-values.
    - Input handImage: Image representing hand contents.
    - Input targImage: Image representing action and current sensed volume.
    - Input flags: State flags, (inHandBit, episodeTime).
    - Input actions: List of candidate actions in the agent's encoding.
    - Return values: Values, one for each action, approximating the q-values.
    '''
    
    idx = 0 if self.IsGraspImage(image) else 1
    dummyIn = zeros((1, 1, 4), dtype='int32')
    values = self.Q[idx][1].predict([array([image]), dummyIn])
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
    
    in1Shape = (self.imP, self.imP, 1) if graspNetwork else (self.imP, self.imP, 2)
      
    in1 = keras.Input(shape=in1Shape, dtype=tensorflow.float32)
    in2 = keras.Input(shape=(1, 4), dtype=tensorflow.int32)
    h1 = keras.layers.Conv2D(params["conv1Outputs"][self.level], \
      kernel_size=params["conv1KernelSize"][self.level], \
      strides=params["conv1Stride"][self.level], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(in1)
    h1 = keras.layers.Conv2D(params["conv2Outputs"][self.level], \
      kernel_size=params["conv2KernelSize"][self.level], \
      strides=params["conv2Stride"][self.level], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    h1 = keras.layers.Conv2D(params["conv3Outputs"][self.level], \
      kernel_size=params["conv3KernelSize"][self.level], \
      strides=params["conv3Stride"][self.level], padding="same", activation="relu", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    h1 = keras.layers.Conv2D(params["conv4Outputs"][self.level], \
      kernel_size=params["conv4KernelSize"][self.level], \
      strides=params["conv4Stride"][self.level], padding="same", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    h1 = keras.layers.Conv2D(params["conv5Outputs"][self.level], \
      kernel_size=params["conv5KernelSize"][self.level], \
      strides=params["conv5Stride"][self.level], padding="same", \
      kernel_regularizer=keras.regularizers.l2(weightDecay))(h1)
    h2 = keras.layers.Lambda(lambda inputs: \
      tensorflow.gather_nd(inputs[0], inputs[1]))([h1, in2])
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
    
    inputs1 = []; inputs2 = []; labels = []    
    
    for i in xrange(2):
      input1 = []; input2 = []; lbl = []
      for d in self.experiences[i]:
        input1.append(d[0]) # image
        index = zeros((1, 4), dtype='int32')
        index[0, 1:] = d[1]
        input2.append(index) # index
        lbl.append(d[2]) # return
      inputs1.append(array(input1))
      inputs2.append(array(input2))
      labels.append(array(lbl))

    return inputs1, inputs2, labels

  def LoadExperienceDatabase(self):
    '''Loads the experience database to file.'''

    path = os.getcwd() + "/tensorflow/" + self.modelFolder + "/experiences_level_" + \
      str(self.level) + ".pickle"
    self.experiences = pickle.load(open(path, "rb"))
    print("Loaded database " + path + ".")

  def LoadQFunction(self):
    '''Loads the network model and weights from the specified file name.'''
    
    directory = os.getcwd() + "/tensorflow/" + self.modelFolder
    
    for i in xrange(2):
      act = "grasp" if i == 0 else "place"
      path = directory + "/q_level_" + str(self.level) + "_" + act + ".h5"
      self.Q[i][0].load_weights(path)
      print("Loaded Q-function " + path + ".")

  def PlotImages(self, o, a):
    '''Produces plots of the observation image.'''

    # Setup

    fig = pyplot.figure()
    It = o[:, :, 0]
    Ih = zeros(It.shape) if self.IsGraspImage(o) else o[:, :, 1]

    # Plot target image

    Ir = copy(It)
    Ig = copy(It)
    Ib = copy(It)

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
    
  def PlotImagesContrastAdjusted(self, o, a):
    '''Same as PlotImages except with adjusted contrast for improved viewability'''

    # Setup

    fig = pyplot.figure()
    It = o[:, :, 0] / max(o[:, :, 0])
    Ih = zeros(It.shape) if self.IsGraspImage(o) else o[:, :, 1] / max(o[:, :, 1])

    # Plot target image

    Ir = copy(It)
    Ig = copy(It)
    Ib = copy(It)

    # draw the selection area
    halfWidth = (It.shape[0] * (self.selW / self.imW)) / 2.0
    middle = It.shape[0] / 2.0
    start = int(round(middle - halfWidth))
    end = int(round(middle + halfWidth))
    pixels = arange(start, end + 1)
    if start >= 0 and end < It.shape[0]:
      Ir[start, pixels] = 0.0
      Ir[end, pixels] = 0.0
      Ir[pixels, start] = 0.0
      Ir[pixels, end] = 0.0
      Ig[start, pixels] = 0.0
      Ig[end, pixels] = 0.0
      Ig[pixels, start] = 0.0
      Ig[pixels, end] = 0.0
      Ib[start, pixels] = 1.0
      Ib[end, pixels] = 1.0
      Ib[pixels, start] = 1.0
      Ib[pixels, end] = 1.0

    # draw robot's selection
    xh = self.actionsInHandFrame[a]
    xi = round(((xh[0:2] * self.imP) / self.imW) + ((self.imP - 1.0) / 2.0)).astype('int32')
    cross = [(xi[0], xi[1]), (xi[0] + 1, xi[1]), (xi[0] - 1, xi[1]), (xi[0] + 2, xi[1]),
      (xi[0] - 2, xi[1]), (xi[0], xi[1]+ 1), (xi[0], xi[1] - 1), (xi[0], xi[1] + 2),
      (xi[0], xi[1] - 2)]
    for p in cross:
      if p[0] >= 0 and p[1] < self.imP and p[1] >= 0 and p[1] < self.imP:
        Ir[p[0], p[1]] = 1
        Ig[p[0], p[1]] = 0
        Ib[p[0], p[1]] = 0

    # show image
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
      self.Q[i][0].save_weights(path)
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
    
    idx = 0 if self.IsGraspImage(image) else 1
    epsilon = 0.0 if unbias else maximum(self.epsilonMin, \
      1.0 - float(len(self.experiences[idx])) / self.maxExperiences)
    
    if rand() < epsilon:
      bestIdx = randint(self.nActionSamples)
      bestIdx = unravel_index(bestIdx, self.outputShape)
      bestValue = float('NaN')
    else:
      values = self.EvaluateActions(image)
      bestIdx = unravel_index(argmax(values), self.outputShape)
      bestValue = values[bestIdx]
      #self.PlotValues(actions, values)

    return bestIdx, bestValue, epsilon

  def UpdateQFunction(self, inputs1, inputs2, labels):
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
      inputs1[i] = inputs1[i][idxs]
      inputs2[i] = inputs2[i][idxs]
      labels[i] = labels[i][idxs]
      # add batch index
      for j in xrange(inputs2[i].shape[0]):
        inputs2[i][j, 0, 0] = j % self.batchSize
      # fit
      history = self.Q[i][0].fit([inputs1[i], inputs2[i]], labels[i], \
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
    inputs1, inputs2, labels = self.LabelDataMonteCarlo()

    return self.UpdateQFunction(inputs1, inputs2, labels)
