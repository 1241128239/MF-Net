import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalEnhancementModule(nn.Module):
    def __init__(self, in_channels, num_patches, embedding_dim):
        super(LocalEnhancementModule, self).__init__()

        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        # Embedding functions
        self.theta = nn.Linear(in_channels, self.embedding_dim)
        self.f = nn.Linear(in_channels, self.embedding_dim)
        self.g = nn.Linear(in_channels, in_channels)

        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        patch_height = height // self.num_patches
        patch_width = width // self.num_patches

        # Split input feature maps into patches
        patches = x.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 1, 3, 4)

        # Calculate weight matrix for each patch
        theta_patches = self.theta(patches)  # [batch_size, num_patches*num_patches, embedding_dim]
        f_patches = self.f(patches)  # [batch_size, num_patches*num_patches, embedding_dim]
        weights = torch.softmax(torch.matmul(theta_patches, f_patches.permute(0, 2, 1)), dim=-1)

        # Transform each patch using g and enhance with weight matrix
        g_patches = self.g(patches)  # [batch_size, num_patches*num_patches, channels, patch_height, patch_width]
        enhanced_patches = weights.unsqueeze(2) * g_patches

        # Concatenate enhanced patches and reshape
        enhanced_patches = enhanced_patches.permute(0, 2, 1, 3, 4).contiguous()
        enhanced_patches = enhanced_patches.view(batch_size, -1, height, width)

        # Multiply by scale and add to input feature maps
        enhanced_feature_maps = self.scale * enhanced_patches + x

        return enhanced_feature_maps
