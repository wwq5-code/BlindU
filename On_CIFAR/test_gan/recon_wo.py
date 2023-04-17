reconstructor = resnet18(3, 3 * 32 * 32)
reconstructor = reconstructor.to(args.device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=0.001)

reconstruction_function = nn.MSELoss(size_average=False)
final_round_mse = []
for epoch in range(init_epoch, init_epoch + args.num_epochs+40):
    vibi.train()
    step_start = epoch * len(dataloader_erase)
    # for step, (x, y) in enumerate(dataloader_erase, start=step_start):
    for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_dp_sampled):
        x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
        x2, y2 = x2.to(args.device), y2.to(args.device)
        if args.dataset == 'MNIST':
            x = x.view(x.size(0), -1)
        logits_z, logits_y, x_hat, mu, logvar = vibi(x2, mode='forgetting')  # (B, C* h* w), (B, N, 10)

        logits_z = logits_z.view(logits_z.size(0), 3, 7, 7)
        x_hat = torch.sigmoid(reconstructor(logits_z))
        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss
        loss = BCE/(x.size(0) * 32 * 32 * 3) * 20

        optimizer_recon.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer_recon.step()
        if epoch == args.num_epochs - 1:
            final_round_mse.append(BCE.item())
    if epoch != 0:
        print("epoch", epoch, " loss", loss.item(), 'BCE', BCE.item())
print("final_round mse", np.mean(final_round_mse))

#########

for step, (x, y) in enumerate(dataloader_dp_sampled):
    x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
    if args.dataset == 'MNIST':
        x = x.view(x.size(0), -1)
    logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
    logits_z = logits_z.view(logits_z.size(0), 3, 7, 7)
    x_hat = torch.sigmoid(reconstructor(logits_z))
    x_hat = x_hat.view(x_hat.size(0), -1)
    x = x.view(x.size(0), -1)
    break

x_hat_cpu = x_hat[18].cpu().data
x_hat_cpu = x_hat_cpu.clamp(0, 1)
x_hat_cpu = x_hat_cpu.view(1, 3, 32, 32)
grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
image_rec= np.transpose(grid.numpy(), (1, 2, 0))
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()
x_cpu = x[18].cpu().data
x_cpu = x_cpu.clamp(0, 1)
x_cpu = x_cpu.view(1, 3, 32, 32)
grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
image_org = np.transpose(grid.numpy(), (1, 2, 0))
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()


plt.imsave('cifar_image_beta_0001_wo.png', image_rec)
plt.imsave('cifar_image_beta_0001_org_wo.png', image_org)

import matplotlib.image as mpimg
# Load the image
image = mpimg.imread('cifar_image_beta_0001_wo.png')

# Display the image
plt.imshow(image)
plt.show()