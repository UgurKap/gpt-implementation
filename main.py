import torch
import torch.nn as nn
from torch.amp import autocast

torch.set_float32_matmul_precision('high')

def main():
    device = torch.device("cuda")

    # Hyper-parameters
    input_size = 1
    output_size = 1
    num_epochs = 200
    learning_rate = 1e-3

    # Toy dataset
    x_train = torch.FloatTensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                        [9.779], [6.182], [7.59], [2.167], [7.042], 
                        [10.791], [5.313], [7.997], [3.1]]).to(device)

    y_train = torch.FloatTensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                        [3.366], [2.596], [2.53], [1.221], [2.827], 
                        [3.465], [1.65], [2.904], [1.3]]).to(device)

    # Linear regression model
    model = torch.compile(nn.Linear(input_size, output_size).to(device))

    # Loss and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    predicted = model(x_train)
    print(f"MAE: {torch.abs(predicted - y_train).mean().item()}")



if __name__ == "__main__":
    main()