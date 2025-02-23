import torch
import numpy as np
import matplotlib.pyplot as plt

# input: image: 3*128*128, action: 1. output: 2
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(3*128*128+1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )
    
    def forward(self, img, a):
        img = img.view(img.size(0), -1)
        x = torch.cat([img, a], dim=1)
        x = self.fc_layer(x)
        return x

images = np.load("data/images.npy")
actions = np.load("data/actions.npy")
positions_after = np.load("data/positions_after.npy")
images = images / 255.0 # normalize images

# prepare data
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
    mlp_model = MLP().to(device)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0001)
    ## loss function: distance between predicted positions and true positions
    criterion = torch.nn.MSELoss()
    num_epochs = 200
    mlp_training_losses = []
    mlp_validation_losses = []
    for epoch in range(num_epochs):
        # training
        running_loss = 0.0
        for i, (images, actions, positions_after) in enumerate(train_loader):
            images = images.to(device)
            actions = actions.to(device)
            positions_after = positions_after.to(device)

            mlp_optimizer.zero_grad()
            outputs = mlp_model(images, actions)
            loss = criterion(outputs, positions_after)
            loss.backward()
            mlp_optimizer.step()
            running_loss += loss.item()*images.size(0)

        epoch_loss = running_loss/len(train_loader.dataset)
        mlp_training_losses.append(epoch_loss)
        print(epoch, epoch_loss)
        
        # validation
        with torch.no_grad():
            outputs = mlp_model(valid_images.to(device), valid_actions.to(device))
            valid_loss = criterion(outputs, valid_positions_after.to(device))
            mlp_validation_losses.append(valid_loss.item())
            if mlp_validation_losses[-1] == min(mlp_validation_losses):
                print("Saving best model, epoch: ", epoch)
                torch.save(mlp_model.state_dict(), "hw1_1.pth")
    return mlp_training_losses

def test():
    mlp_model = MLP()
    mlp_model.load_state_dict(torch.load("hw1_1.pth", weights_only=True))
    criterion = torch.nn.MSELoss()

    mlp_test_losses = []  

    with torch.no_grad():
        for i, (images, actions, positions_after) in enumerate(test_loader):
            mlp_outputs = mlp_model(images, actions)
            mlp_loss = criterion(mlp_outputs, positions_after)
            mlp_test_losses.append(mlp_loss.item())

    ## write to file
    with open("results/mlp_test_results.txt", "w") as f:
        f.write("MLP test loss: " + str(mlp_test_losses) + "\n")

def plot_loss(mlp_training_losses):
    plt.figure()
    plt.plot(mlp_training_losses, label='MLP')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training Loss of MLP Model')
    plt.tight_layout()
    plt.savefig('results/mlp_training_loss.png')

mlp_training_losses = train()
plot_loss(mlp_training_losses)
test()

