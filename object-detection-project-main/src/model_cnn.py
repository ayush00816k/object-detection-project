import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.network = nn.Sequential(
            # 1️⃣ First Convolution Block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # 3 = input channels(RGB) ,
            #  32 = output channels(filters) ,
            #  kernel size = (3x3)filter , 
            # stride = how many pixels the filter moves each time.
            # padding = adding padding as per given 
            nn.BatchNorm2d(32),
            # normalizing the result after the convolution layer (32 is because of the prev convolutional layer has 32 output channels)
            nn.ReLU(),
            # Helps the model learn faster, better, and more complex patterns like for no. (if negative make it 0 , if positive keep it as it is )
            nn.MaxPool2d(2, 2),
            # Reduces image size by half and extracts only the strongest features


            # 2️⃣ Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 3️⃣ Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 4️⃣ Fourth Convolution Block (NEW)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Regularization + Classification Layers (Final layer that predicts the class)
            nn.Dropout(0.3),# Dropout randomly 40% of neurons during training.
            nn.Flatten(), # Converts the 3D feature map into a 1D vector
            nn.Linear(256 * 8 * 8, 256),  # assumes 128×16×16 feature map (layer that connects every input to every output.)
            nn.ReLU(), # again relu for more regularization and non-linearity
            nn.Dropout(0.3), # Another dropout for extra protection against overfitting.
            nn.Linear(256, num_classes) # This layer produces the final predictions.
        )

    def forward(self, x):
        return self.network(x)


# Deeper layers learn MORE complex features (hence the input/output channels are increasedafter every blocks)