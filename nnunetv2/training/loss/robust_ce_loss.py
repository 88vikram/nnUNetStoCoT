import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        ce_loss = super().forward(input, target.long())
        print(ce_loss)
        return ce_loss
    

class StoCoTRobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
       
    def repeat_and_truncate(self,tensor, target_shape, missing_dim=None):
        
        if missing_dim is not None:
            tensor = tensor.unsqueeze(missing_dim)

        device = tensor.device  # Ensure we operate on the correct device
        # Calculate repeat_factors
        repeat_factors = []
        for i, current_dim in enumerate(tensor.shape):
            # Calculate how many times the tensor needs to be repeated in each dimension
            repeat_factor = target_shape[i] // current_dim
            if target_shape[i] % current_dim != 0:
                repeat_factor += 1  # Ensure full coverage by repeating one extra time if needed
            repeat_factors.append(repeat_factor)
        
        repeated_tensor_full = tensor.repeat(*repeat_factors).to(device)

        # Slice (truncate) if repeated tensor exceeds target shape
        slices = tuple(slice(0, min(repeated_tensor_full.shape[i], target_shape[i])) for i in range(len(target_shape)))
        final_repeated_tensor=repeated_tensor_full[slices].to(device)

        return final_repeated_tensor

    def mask_probabilities(self,x,y,final_stochastic_thresholds):        

        class_dim=1
        probabilities = F.softmax(x, dim=class_dim)
        class_probabilities = torch.gather(probabilities,class_dim,y.unsqueeze(class_dim).long())
        mask = torch.ge(class_probabilities, final_stochastic_thresholds[:,0]).type(torch.cuda.FloatTensor)
        mask = mask.expand(x.shape)

        return mask

    def forward(self, input1: Tensor,input2: Tensor, target: Tensor,stochastic_thresholds:Tensor):
        if target.ndim == input1.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        final_stochastic_thresholds=self.repeat_and_truncate(stochastic_thresholds,input1.shape,1)
        stocot_mask1 = self.mask_probabilities(input1,target,final_stochastic_thresholds).to(torch.bool)
        stocot_mask2 = self.mask_probabilities(input2,target,final_stochastic_thresholds).to(torch.bool)
        
        loss_object=nn.CrossEntropyLoss(reduction='none')

        if input1.ndim != target.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        if input1.shape == target.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = target
        else:
            y_onehot = torch.zeros(input1.shape, device=input1.device, dtype=torch.bool)
            y_onehot.scatter_(1, target.long(), 1)


        loss1=(loss_object.forward(input1, y_onehot.float()))*stocot_mask2
        loss2=(loss_object.forward(input2, y_onehot.float()))*stocot_mask1

        ce_loss1=loss1.sum() / stocot_mask2.sum()
        ce_loss2=loss2.sum() / stocot_mask1.sum()
        
        return ce_loss1,ce_loss2


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
