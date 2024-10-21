import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
import numpy as np
from dataset import RoboticDataset
from dataset_sequential import SequentialRoboticDataset
from network import RobotPolicyModel

slices = [0, 1, 2, 3, 4, 5, 6] # non-zero action indices

def main(args):
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = SequentialRoboticDataset(root_dir=args.root_dir, n_obs=args.n_obs, n_pred=args.n_pred, interval=args.interval, obs_interval=args.obs_interval, test=False)
    test_dataset = SequentialRoboticDataset(root_dir=args.root_dir, n_obs=args.n_obs, n_pred=args.n_pred, interval=args.interval, obs_interval=args.obs_interval, test=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model, Loss, and Optimizer
    model = RobotPolicyModel(n_obs=args.n_obs, n_pred=args.n_pred, seperate_encoder=args.seperate_encoder, status_conditioned=args.status_conditioned, bottleneck_dim=args.bottleneck_dim).to(device)
    model = torch.nn.DataParallel(model)

    if args.loss_function == 'mse':
        criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
    elif args.loss_function == 'l1':
        criterion = torch.nn.L1Loss()  # L1 Loss (Mean Absolute Error)
    
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # TensorBoard``
    log_dir = os.path.join('runs', f'n_obs_{args.n_obs}_n_pred_{args.n_pred}_interval_{args.interval}_batch_size_{args.batch_size}_lr_{args.learning_rate}_loss_{args.loss_function}_seperate_encoder_{args.seperate_encoder}_status_conditioned_{args.status_conditioned}_obs_interval_{args.obs_interval}_bottleneck-dim-{args.bottleneck_dim}_vae_weight-{args.vae_weight}_weight-decay-{args.weight_decay}')
    writer = SummaryWriter(log_dir)

    # Checkpoints
    checkpoint_dir = os.path.join('checkpoints', f'n_obs_{args.n_obs}_n_pred_{args.n_pred}_interval_{args.interval}_batch_size_{args.batch_size}_lr_{args.learning_rate}_loss_{args.loss_function}_seperate_encoder_{args.seperate_encoder}_status_conditioned_{args.status_conditioned}__obs_interval_{args.obs_interval}_bottleneck-dim-{args.bottleneck_dim}_vae_weight-{args.vae_weight}_weight-decay-{args.weight_decay}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_losses = [float('inf')] * 3

    global vae_weight, bottleneck_dim
    vae_weight = args.vae_weight
    bottleneck_dim = args.bottleneck_dim

    def train(model, train_loader, criterion, optimizer, epoch, portion=0.1):
        global vae_weight, bottleneck_dim

        print('Bottleneck Dim:', bottleneck_dim, 'VAE Weight:', vae_weight)

        model.train()
        running_loss = 0.0

        with tqdm(total=int(len(train_loader) * portion), desc=f'Epoch {epoch+1}/{args.num_epochs}', unit='batch') as pbar:
            for i, batch in enumerate(train_loader):
                if i >= len(train_loader) * portion:
                    break
                optimizer.zero_grad()

                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Get ground truth future actions
                action_gt = batch['action'][:, args.n_obs:]

                # Truncate observation steps as input
                batch['action'] = batch['action'][:, :args.n_obs]

                if bottleneck_dim is None:
                    predicted_action = model(batch)
                    loss = criterion(predicted_action[:, :, :], action_gt[:, :, slices])
                else:
                    predicted_action, loss_vae = model(batch)
                    loss_vae = loss_vae.mean()
                    loss = criterion(predicted_action[:, :, :], action_gt[:, :, slices]) + vae_weight * loss_vae

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'Loss': loss.item(), 'Loss VAE': 0.0 if bottleneck_dim is None else loss_vae.item()})
                pbar.update()

                writer.add_scalar('Batch Loss', loss.item(), epoch * len(train_loader) * portion + i)
                # writer.add_scalar('Batch Loss VAE', loss_vae.item(), epoch * len(train_loader) + i)

        epoch_loss = running_loss / (len(train_loader) * portion)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {epoch_loss:.4f}')

    def evaluate(model, test_loader, criterion, epoch):
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Get ground truth future actions
                action_pred = batch['action'][:, args.n_obs:]

                # Truncate observation steps as input
                batch['action'] = batch['action'][:, :args.n_obs]

                predicted_action = model(batch)

                # Calculate loss
                loss = criterion(predicted_action[:, :, :], action_pred[:, :, slices])
                test_loss += loss.item()

        test_loss /= len(test_loader)
        writer.add_scalar('Validation Loss', test_loss, epoch)
        print(f'Test Loss: {test_loss:.4f}')
        return test_loss

    # Training Loop
    for epoch in range(args.num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        val_loss = evaluate(model, test_loader, criterion, epoch)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        # if dataparallel, save module weights
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)

        # Save checkpoints
        if val_loss < max(best_losses):
            best_losses[best_losses.index(max(best_losses))] = val_loss
            best_losses.sort()
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

    print("Training and evaluation completed.")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Robot Policy Model')
    parser.add_argument('--root_dir', type=str, default='../fifish-vevo/rov_control/data_0806', help='Path to the dataset')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--n_obs', type=int, default=1, help='Number of observation steps')
    parser.add_argument('--n_pred', type=int, default=8, help='Number of prediction steps')
    parser.add_argument('--bottleneck_dim', type=int, default=None, help='Number of dimension in policy')
    parser.add_argument('--vae_weight', type=float, default=0.0, help='Image size')
    parser.add_argument('--interval', type=int, default=100, help='Interval between steps in ms')
    parser.add_argument('--obs_interval', type=int, default=100, help='Interval between steps in ms for observation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--loss_function', type=str, choices=['mse', 'l1'], default='mse', help='Loss function (mse or l1)')
    parser.add_argument('--seperate_encoder', action='store_true', help='Use seperate encoder for ROV images')
    parser.add_argument('--status_conditioned', action='store_true', help='Condition on status data')
    args = parser.parse_args()

    main(args)


'''
python train.py --root_dir ../fifish-vevo/rov_control/data_train --num_epochs 200 --n_obs 2 --n_pred 2 --interval 100 --batch_size 32 --learning_rate 0.0001 --weight_decay 0.1 --loss_function mse --obs_interval 300 --seperate_encoder
python train.py --root_dir ../fifish-vevo/rov_control/data_train --num_epochs 200 --n_obs 2 --n_pred 4 --interval 100 --batch_size 32 --learning_rate 0.0001 --weight_decay 0.1 --loss_function mse --obs_interval 300 --seperate_encoder
python train.py --root_dir ../fifish-vevo/rov_control/data_train --num_epochs 200 --n_obs 2 --n_pred 8 --interval 100 --batch_size 32 --learning_rate 0.0001 --weight_decay 0.1 --loss_function mse --obs_interval 300 --seperate_encoder
'''