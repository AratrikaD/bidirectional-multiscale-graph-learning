from abc import ABC, abstractmethod
import os
from dataclasses import dataclass


class DirectoryConfig(object):
    ROOT = os.path.dirname(__file__)
    MODELS = os.path.join(ROOT, 'src/models')
    DATA = os.path.join(ROOT, 'housing-data')
    AZUREML = os.path.join(ROOT, '.azureml')
 