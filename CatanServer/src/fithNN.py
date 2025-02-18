from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import catanData
import abstractCatanBot
from compiledCordinateSystem import compiledHexIndex, compiledEdgeIndex, compiledCornerIndex
import random
