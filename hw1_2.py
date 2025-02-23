import torch
import numpy as np
import matplotlib.pyplot as plt

# input: image: 3*128*128, action: 1. output: 2
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128*16*16+1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2)
        )
    
    def forward(self, img, a):
        x = self.conv_layer(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, a], dim=1)
        x = self.fc_layer(x)
        return x

images = np.load("data/images.npy")
actions = np.load("data/actions.npy")
positions_after = np.load("data/positions_after.npy")

images = images / 255.0

dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), torch.tensor(positions_after, dtype=torch.float32))
train_size = int(0.8*len(images))
valid_size = int(0.1*len(images))
test_size = len(images) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_images, valid_actions, valid_positions_after = valid_dataset[:]
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train():
    cnn_model = CNN().to(device)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)
    ## loss function: distance between predicted positions and true positions
    criterion = torch.nn.MSELoss()
    num_epochs = 500
    cnn_training_losses = []
    cnn_validation_losses = []
    for epoch in range(num_epochs):
        # training
        running_loss = 0.0
        for i, (images, actions, positions_after) in enumerate(train_loader):
            images = images.to(device)
            actions = actions.to(device)
            positions_after = positions_after.to(device)

            cnn_optimizer.zero_grad()
            outputs = cnn_model(images, actions)
            loss = criterion(outputs, positions_after)
            loss.backward()
            cnn_optimizer.step()
            running_loss += loss.item()*images.size(0)

        epoch_loss = running_loss/len(train_loader.dataset)
        cnn_training_losses.append(epoch_loss)
        print(epoch, epoch_loss)
        
        # validation
        with torch.no_grad():
            outputs = cnn_model(valid_images.to(device), valid_actions.to(device))
            valid_loss = criterion(outputs, valid_positions_after.to(device))
            cnn_validation_losses.append(valid_loss.item())
            if cnn_validation_losses[-1] == min(cnn_validation_losses):
                print("Saving best model, epoch: ", epoch)
                torch.save(cnn_model.state_dict(), "save/cnn_best_model.pth")
    return cnn_training_losses

def test():
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load("save/cnn_best_model.pth"))
    criterion = torch.nn.MSELoss()

    cnn_test_losses = []  

    with torch.no_grad():
        for i, (images, actions, positions_after) in enumerate(test_loader):
            cnn_outputs = cnn_model(images, actions)
            cnn_loss = criterion(cnn_outputs, positions_after)
            cnn_test_losses.append(cnn_loss.item())

    ## write to file
    with open("results/cnn_test_results.txt", "w") as f:
        f.write("CNN test loss: " + str(cnn_test_losses) + "\n")

def plot_loss(cnn_training_losses):
    plt.figure()
    plt.plot(cnn_training_losses, label='CNN')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training Loss of CNN Model')
    plt.savefig('results/cnn_training_loss.png')

cnn_training_losses = train()
plot_loss(cnn_training_losses)
test()





