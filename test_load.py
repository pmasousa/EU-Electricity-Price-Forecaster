import os, torch, pickle
from darts.models import TFTModel

print('loading')
model = TFTModel.load('models/tft_model.pt', map_location='cpu', weights_only=False)
print('success')
