import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        imgSize = 28
        nrConvFilter = 3
        kernelSize = 5
        fcInputSize = (imgSize - 2 * (kernelSize // 2)) * (imgSize - 2 * (kernelSize // 2)) * nrConvFilter // (2 * 2)

        self.convLayer = nn.Conv2d(in_channels=1, out_channels=nrConvFilter, kernel_size=kernelSize)
        self.poolLayer = nn.MaxPool2d(2)
        self.fcLayer = nn.Linear(fcInputSize, 10)

    def forward(self, x):
        x = self.convLayer(x)
        x = self.poolLayer(x)
        x = torch.relu(x)
        x = x.view([1, -1])
        x = self.fcLayer(x)
        return F.log_softmax(x, dim=1)


train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

network = ConvolutionalNetwork().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.001)

nrEpochs = 10
for epoch in range(nrEpochs):
    for data in trainset:
        imgs, values = data
        for img, value in zip(imgs, values):
            img = img.to(device)
            value = value.to(device)

            optimizer.zero_grad()
            predicted = network(img.unsqueeze(0))
            loss = F.nll_loss(predicted, value.unsqueeze(0))
            loss.backward()
            optimizer.step()

    print('Epoch:', epoch + 1, 'Loss:', loss.item())

correct = 0
total = 0

for data in testset:
    imgs, values = data
    for img, value in zip(imgs, values):
        img = img.to(device)
        value = value.to(device)

        network.zero_grad()
        output = network(img.unsqueeze(0))

        for index, distribution in enumerate(output):
            if torch.argmax(distribution) == value:
                correct += 1
            total += 1

print("Accuracy: {}%".format(round(correct / total, 3) * 100))
