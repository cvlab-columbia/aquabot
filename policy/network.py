import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from img_encoder import FeatureExtractorCNN
from spatial_softmax import SpatialSoftmaxCNN, SpatialSoftmaxResNet

feature_size = 512
image_size = 224
action_size = 7

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_pred, action_contidioned=False, seperate_encoder=False, status_conditioned=False, bottleneck_dim=None):
        super(PolicyNetwork, self).__init__()
        self.action_contidioned = action_contidioned
        self.seperate_encoder = seperate_encoder
        self.status_conditioned = status_conditioned
        if not seperate_encoder:
            if action_contidioned and status_conditioned:
                in_dim = 4 * n_obs * feature_size + n_obs * 18
            elif status_conditioned and not action_contidioned:
                in_dim = 4 * n_obs * feature_size + n_obs * 3
            elif action_contidioned and not status_conditioned:
                in_dim = 4 * n_obs * feature_size + n_obs * action_size
            else:
                in_dim = 4 * n_obs * feature_size
        else:
            if action_contidioned and status_conditioned:
                in_dim = 2 * n_obs * feature_size + n_obs * 18
            elif status_conditioned and not action_contidioned:
                in_dim = 2 * n_obs * feature_size + n_obs * 3
            elif action_contidioned and not status_conditioned:
                in_dim = 2 * n_obs * feature_size + n_obs * action_size
            else:
                in_dim = 2 * n_obs * feature_size
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.bottleneck_dim = bottleneck_dim

        if bottleneck_dim is None:
            self.fc1 = nn.Linear(in_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, n_pred * action_size)
        else:
            self.fc1 = nn.Linear(in_dim, 64)
            self.fc2 = nn.Linear(64, 2 * bottleneck_dim)
            self.fc3 = nn.Linear(bottleneck_dim, 64)
            self.fc4 = nn.Linear(64, n_pred * action_size)
    
    def forward(self, in_feature):
        if self.bottleneck_dim is None:
            x = in_feature
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = self.fc3(x)
            return x
        else:
            x = in_feature
            x = F.leaky_relu(self.fc1(x))

            # variational autoencoder (reparameterization trick)
            mu, logvar = self.fc2(x).chunk(2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = mu + eps * std
            loss_vae = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            x = F.leaky_relu(self.fc3(x))
            x = self.fc4(x)
            return x, loss_vae

class RobotPolicyModel(nn.Module):
    def __init__(self, n_obs, n_pred, action_contidioned=False, seperate_encoder=False, status_conditioned=False, bottleneck_dim=None):
        super(RobotPolicyModel, self).__init__()
        self.action_contidioned = action_contidioned
        self.status_conditioned = status_conditioned
        self.bottleneck_dim = bottleneck_dim
        
        if seperate_encoder:
            self.rov_mount_encoder = SpatialSoftmaxResNet()
            self.rov_main_encoder = SpatialSoftmaxResNet()
        else:
            self.img_encoder = SpatialSoftmaxResNet()
        self.seperate_encoder = seperate_encoder
        self.policy_network = PolicyNetwork(n_obs, n_pred, action_contidioned, seperate_encoder, status_conditioned, bottleneck_dim)
        self.n_obs = n_obs
        self.n_pred = n_pred
    
    def forward(self, input):
        batch_size = input["cctv_left"].size(0)
        
        # Concatenate image inputs
        cctv_left = input["cctv_left"].view(batch_size * self.n_obs, 3, image_size, image_size)
        cctv_right = input["cctv_right"].view(batch_size * self.n_obs, 3, image_size, image_size)
        rov_main = input["rov_main"].view(batch_size * self.n_obs, 3, image_size, image_size)
        rov_mount = input["rov_mount"].view(batch_size * self.n_obs, 3, image_size, image_size)
        
        # Encode images
        if not self.seperate_encoder:
            features_left = self.img_encoder(cctv_left).view(batch_size, self.n_obs, feature_size)
            features_right = self.img_encoder(cctv_right).view(batch_size, self.n_obs, feature_size)
            features_main = self.img_encoder(rov_main).view(batch_size, self.n_obs, feature_size)
            features_mount = self.img_encoder(rov_mount).view(batch_size, self.n_obs, feature_size)
            img_features = torch.cat([features_left, features_right, features_main, features_mount], dim=1)
        else:
            features_main = self.rov_main_encoder(rov_main).view(batch_size, self.n_obs, feature_size)
            features_mount = self.rov_mount_encoder(rov_mount).view(batch_size, self.n_obs, feature_size)
            img_features = torch.cat([features_main, features_mount], dim=1)
        
        img_features = img_features.view(batch_size, -1)  # Flatten to [batch_size, 4*n_obs*feature_size]
        
        if self.action_contidioned and self.status_conditioned:
            # Concatenate action with status
            action_status = torch.cat([input['action'], input["status"]], dim=2)  # [batch_size, n_obs, action_size + 3]
            action_status = action_status.view(batch_size, -1)  # Flatten to [batch_size, n_obs * 18]
        elif self.status_conditioned and not self.action_contidioned:
            # Concatenate status
            action_status = input["status"].view(batch_size, -1)
        elif self.action_contidioned and not self.status_conditioned:
            # Concatenate action
            action_status = input["action"].view(batch_size, -1)
        
        if self.action_contidioned or self.status_conditioned:
            # Concatenate image features with action and status
            combined_features = torch.cat([img_features, action_status], dim=1)  # [batch_size, 4*n_obs*feature_size + n_obs*18]
        else:
            combined_features = img_features
        
        if self.bottleneck_dim is None:
            # Predict future actions
            predicted_action = self.policy_network(combined_features)  # [batch_size, n_pred*action_size]
            predicted_action = predicted_action.view(batch_size, self.n_pred, action_size)  # Reshape to [batch_size, n_pred, action_size]
            return predicted_action
        else:
            # Predict future actions and get VAE loss
            predicted_action, loss_vae = self.policy_network(combined_features)
            predicted_action = predicted_action.view(batch_size, self.n_pred, action_size)
        
            return predicted_action, loss_vae

if __name__ == "__main__":
    n_obs = 2
    n_pred = 1
    model = RobotPolicyModel(n_obs, n_pred)
    batch = {
        "cctv_left": torch.randn(2, n_obs, 3, image_size, image_size),
        "cctv_right": torch.randn(2, n_obs, 3, image_size, image_size),
        "rov_main": torch.randn(2, n_obs, 3, image_size, image_size),
        "rov_mount": torch.randn(2, n_obs, 3, image_size, image_size),
        "action": torch.randn(2, n_obs + n_pred, action_size),
        "status": torch.randn(2, n_obs, 3)
    }
    output = model(batch)
    print(output[0].shape)  # Expected output shape: [2, n_pred, action_size]
    print(output[1].shape)  # Expected output shape: [2, n_pred, action_size]