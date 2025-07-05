import torch
import transformers
import fastapi
import streamlit
import pandas as pd
import numpy as np
import gradio
print('Environment setup successful!')
print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import jupyterlab
print('JupyterLab imported successfully')
