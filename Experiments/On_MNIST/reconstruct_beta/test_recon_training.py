reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
reconstructor = reconstructor.to(device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)

reconstruction_function = nn.MSELoss(size_average=False)
final_round_mse = []

for epoch in range(init_epoch, init_epoch + args.num_epochs):
    vibi.train()
    step_start = epoch * len(dataloader_dp_sampled)
    # for step, (x, y) in enumerate(dataloader_dp_sampled, start=step_start):
    step = 0
    for (x, y), (x2, y2) in zip(dataloader_dp_sampled, dataloader_erase):
        step = step+100
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        x2 = x2.view(x2.size(0), -1).to(args.device)
        logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

        x_hat = torch.sigmoid(reconstructor(logits_z))
        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x2)  # mse loss
        loss = BCE

        optimizer_recon.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer_recon.step()

        if epoch == args.num_epochs - 1:
            final_round_mse.append(BCE.item())
        if step % len(train_loader) % 600 == 0:
            print("loss", loss.item(), 'BCE', BCE.item())

#########

for step, (x, y) in enumerate(dataloader_dp_sampled):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    x = x.view(x.size(0), -1)
    logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
    x_hat = torch.sigmoid(reconstructor(logits_z))
    x_hat = x_hat.view(x_hat.size(0), -1)
    x = x.view(x.size(0), -1)
    break
index = 0
x_hat_cpu = x_hat[index].cpu().data
x_hat_cpu = x_hat_cpu.clamp(0, 1)
x_hat_cpu = x_hat_cpu.view(1, 1, 28, 28)
grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
image_rec= np.transpose(grid.numpy(), (1, 2, 0))
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()
x_cpu = x[index].cpu().data
x_cpu = x_cpu.clamp(0, 1)
x_cpu = x_cpu.view(1, 1, 28, 28)
grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
image_org = np.transpose(grid.numpy(), (1, 2, 0))
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()

plt.imsave('mnist_image_beta_0001_wo.png', image_rec)
plt.imsave('mnist_image_beta_0001_org_wo.png', image_org)

import matplotlib.image as mpimg
# Load the image
image = mpimg.imread('mnist_image_beta_0001_wo.png')

# Display the image
plt.imshow(image)
plt.show()