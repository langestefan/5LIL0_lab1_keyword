import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class KeywordLeNet5Clone(BaseModel):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
        self.fc3 = nn.Linear(184, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.view(x.size(0), 184)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
