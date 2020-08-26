import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
from ntap.models import Model

class CNN(Model):
    def __init__(self,formula,data,activation="relu",dropout = 0.5,loss = "Xentropy",optimizer = "adam",pooling = "max"):
        """"""
        self.activation = "relu"
        self.droput = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.pooling = pooling


    def __parse_formula(self, formula, data):
       """Parse the input and create token"""


    def build(self, ):
        """Build the CNN using Tensorflow libraries, explained in the document"""

    def __buildCNN(self):
        """Depending upon the input parameters, build the CNN"""

