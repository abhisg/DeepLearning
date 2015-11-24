import numpy as np

class DBN:
  def __init__(self, num_visible, hidden_layers,options, learning_rate = 0.1,batches = 1):
    # num_visible : number of visible units
    # hidden_layers: list if number of units for each layer
    # options : list of options for continuous layers. if a layer is binary it 0,else it will be 1
    # learning_rate : learning rate for each RBM

    self.hidden_layers = hidden_layers
    self.options = options
    self.hidden_layers.insert(0,num_visible)
    self.RBMs = []
    self.L = len(self.hidden_layers)
    self.batches = batches
    
    # initialize each RBMs
    for l in xrange(0,self.L - 1):
      r = RBM(num_visible = self.hidden_layers[l], num_hidden = self.hidden_layers[l+1], learning_rate = learning_rate)
      (self.RBMs).append(r)

  def train(self,data,max_epochs = 1000):
    # train the DBN as Bengio et al. suggested in "Greedy Layer-Wise Training of Deep Networks"
    for l in xrange(0,self.L -1):
      vis = data
      for i in range(0,l):
        print "training level",l,"sampling level ",i
        hid = (self.RBMs[i]).run_visible(vis,self.options[i])
        vis = hid
    
      print "input=",vis
      (self.RBMs[l]).train(vis, max_epochs,self.options[l],self.batches)
      print l,"level is trained"

  def run_visible(self,data):
    # run observed data to learn hidden units
    vis = data
    for l in xrange(0,self.L -1):
      hid = (self.RBMs[l]).run_visible(vis,self.options[l+1])
      vis = hid
    return hid

  def run_hidden(self,data):
    # run hidden units to generate visible units
    hid = data
    for l in reversed(range(0,self.L -1)):
      vis = (self.RBMs[l]).run_hidden(hid,self.options[l])
      hid = vis
    return vis

class RBM:

  def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)    
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, opt = 0,batches=1):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.
    """
    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)
    batch_size = num_examples/batches
    
    for epoch in range(max_epochs):
        shuffled = np.random.permutation(data)
        for batch in xrange(batches):
            startidx,endidx = batch*batch_size,min((batch+1)*batch_size,num_examples)
            #print startidx,endidx
            sample = shuffled[startidx:endidx]
            #print sample.shape
            # Clamp to the data and sample from the hidden units. 
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(sample, self.weights)      
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(sample.shape[0], self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(sample.T, pos_hidden_probs)
            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            if opt == 0:
                neg_visible_probs = self._logistic(neg_visible_activations)
            else:
                neg_visible_probs = self._continuousLogistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            # Update weights.
            #print "WEIGHT=",self.weights
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / sample.shape[0])
            error = np.mean((sample - neg_visible_probs) ** 2)
            #print "delta =",self.learning_rate * ((pos_associations - neg_associations) / sample.shape[0])
            #print "Epoch %s batch %s: error is %s" % (epoch, batch, error)

  def run_visible(self, data,opt):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.

    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)


    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)

    if opt == 0:
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Always fix the bias unit to 1.
      # hidden_states[:,0] = 1
    else:
      hidden_probs = self._continuousLogistic(hidden_activations)
      hidden_states[:,:] = hidden_probs

    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data,opt):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    if opt == 0:
      visible_probs = self._logistic(visible_activations)
      visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    else:
      visible_probs = self._continuousLogistic(visible_activations)
      visible_states[:,:] = visible_probs 

    # Turn the visible units on with their specified probabilities.


    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states

  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

  def _continuousLogistic(self, x):
    n,m = x.shape
    U = np.random.rand(n,m)

    return np.log(1 - (U*(1-np.exp(x))))/(x+0.000001)
