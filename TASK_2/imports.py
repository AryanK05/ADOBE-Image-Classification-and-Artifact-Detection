from PIL import Image
import pandas as pd
import requests
from transformers import pipeline
import torch
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, ViTFeatureExtractor, ViTForImageClassification
from sentence_transformers import SentenceTransformer, util
from qwen_vl_utils import process_vision_info
from open_clip import create_model_and_transforms, create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np