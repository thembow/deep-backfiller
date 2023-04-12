from job import Job, Workloads
from cluster import Cluster

import os
import math
import json
import time
import sys
import random
from random import shuffle

import numpy as np
import tensorflow as tf
import scipy.signal

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

import joblib
import os.path as osp
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from tensorflow.python.util import deprecation
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True

from HPCSimPickJobs import *
tf.enable_eager_execution()

"""1. PPO picks a job to run backfill-gym simulates the normal job scheduling and running until there is a backfilling opportunity (You can learn how the normal scheduling works from HPCenv and adapt it to backfill-gym. HPCenv should be finally replaced with backfill-gym.)
2. backfill-gym provides the current observation as state to PPO.
3. PPO picks a job in the waiting queue to backfill.
4. The job is sent to backfill-gym. 
5. backfill-gym backfills the job, if the backfilling opportunity still exists, it provides the current observation to PPO.
6. go back to 3 until there is no backfilling opportunity.
7. go back to 1."""