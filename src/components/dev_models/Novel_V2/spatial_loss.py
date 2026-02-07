import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelOperator(nn.Module):
    def __init__(self, in_channels=1):
        super(SobelOperator, self).__init__()
        # Define Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Register kernels as non-trainable buffers
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.in_channels = in_channels

    def forward(self, x):
        # x shape: (B, 1, H, W) or (B, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # If the input is class indices (integers), convert to float
        x = x.float()

        # Replicate kernels for number of channels if needed (usually 1 for mask)
        weight_x = self.sobel_x.repeat(x.size(1), 1, 1, 1)
        weight_y = self.sobel_y.repeat(x.size(1), 1, 1, 1)

        grad_x = F.conv2d(x, weight_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, weight_y, padding=1, groups=x.size(1))
        
        # Calculate magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Normalize to 0-1 range for binary edge mask
        edge_map = torch.sigmoid(magnitude) 
        
        # Thresholding creates crisp edges from the semantic mask
        return edge_map

class SpatialEdgeLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(SpatialEdgeLoss, self).__init__()
        self.sobel = SobelOperator()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_edge_logits, gt_semantic_mask):
        """
        pred_edge_logits: (B, 1, H, W) - Output from the edge branch
        gt_semantic_mask: (B, H, W) - Ground truth segmentation indices
        """
        # 1. Extract Edges from Ground Truth using Sobel
        with torch.no_grad():
            # Get edges (result is 0 to 1)
            gt_edges = self.sobel(gt_semantic_mask)
            # Binarize GT edges (any gradient implies an edge)
            gt_edges = (gt_edges > 0.1).float()

        # 2. Compute Loss
        # BCE Loss
        bce_loss = self.bce(pred_edge_logits, gt_edges)
        
        # Dice Loss for Edges (Handling class imbalance for thin edges)
        pred_edge_prob = torch.sigmoid(pred_edge_logits)
        intersection = (pred_edge_prob * gt_edges).sum(dim=(2, 3))
        union = pred_edge_prob.sum(dim=(2, 3)) + gt_edges.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss