from torch import nn
from torchvision.models import resnet152,ResNet152_Weights
from transformers.models.informer import InformerPreTrainedModel,InformerConfig,InformerForPrediction,InformerModel,P
from transformers.models.informer import Informer
from transformers import PretrainedConfig
import pandas as pd
config=InformerConfig(
    input_size=21,
    prediction_length=21,
)
model=InformerPreTrainedModel(PretrainedConfig(
    max_length=21,
    min_length=21,
    input_size=21,
))
numerical_data=pd.read_csv('./dataset/Banqiao_2022.csv')