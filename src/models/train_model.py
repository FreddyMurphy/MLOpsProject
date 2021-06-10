def train(model, device, trainloader, epochs=5, print_every=500):
    steps = 0
    running_loss = 0
    model.train()
    for e in range(epochs):
        # Model in training mode, dropout is on
        batch_idx = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            loss = model.training_step((images, labels), batch_idx)
            running_loss += loss.item()

            batch_idx += 1

            if steps % print_every == 0:

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. "
                      .format(running_loss/print_every))
                running_loss = 0
