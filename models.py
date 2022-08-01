from torch import nn
import torchvision


class NeuralNetwork(nn.Module):
    """
    Generic fully-connected network class
    The first argument of the constructor should be the number of input features, and the last should be the number of outputs
    """
    # layers = int layer sizes, starting with the input layer
    def __init__(self, *layers):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        stack = []
        for i in range(len(layers) - 1):
            stack.extend([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()])

        self.linear_relu_stack = nn.Sequential(*stack[:-1])

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TransferModel18(nn.Module):
    """
    ResNet18 transfer learning model
    By default we set the fully-connected classifier layer to a single 512x2 layer
    """
    def __init__(self, pretrained=True, freeze=True, device='cpu'):
        super(TransferModel18, self).__init__()

        if pretrained:
            self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.default).to(device)
        else:
            self.model = torchvision.models.resnet18().to(device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Fully-connected layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2, bias=True, device=device),
        )

        for layer in self.model.fc:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.model(x).squeeze()


class TransferModel50(nn.Module):
    """
    ResNet50 transfer learning model
    """
    def __init__(self, pretrained=True, freeze=True, device='cpu'):
        super(TransferModel50, self).__init__()

        if pretrained:
            self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.default).to(device)
        else:
            self.model = torchvision.models.resnet50().to(device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Fully-connected layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2, bias=True, device=device),
        )

        for layer in self.model.fc:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.model(x).squeeze()


