import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label and single-label classification tasks.

    Implementation from: https://github.com/Alibaba-MIIL/ASL
    
    Asymmetric Loss from: "Asymmetric Loss For Multi-Label Classification"
    https://openaccess.thecvf.com/content/ICCV2021/papers/Ben-Baruch_Asymmetric_Loss_For_Multi-Label_Classification_ICCV_2021_paper.pdf
    
    Args:
        gamma_neg (float): Negative focusing parameter. Default is 4.
        gamma_pos (float): Positive focusing parameter. Default is 1.
        clip (float, optional): Asymmetric clipping value for negative probabilities. Default is 0.05.
        eps (float, optional): Small epsilon value for numerical stability. Default is 1e-8.
        reduction (str, optional): Specifies the reduction method: 'none' | 'mean' | 'sum'. Default is 'mean'.
        task_type (str, optional): Type of task: 'multi-label' or 'single-label'. Default is 'multi-label'.
        disable_torch_grad_focal_loss (bool, optional): Whether to disable gradient computation during focal loss calculation. Default is True.
        optimized (bool, optional): Whether to use optimized version with inplace operations (only for multi-label). Default is False.
    """

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        reduction="mean",
        task_type="multi-label",
        disable_torch_grad_focal_loss=True,
        optimized=False,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.task_type = task_type
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.optimized = optimized
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []

        # For optimized version: pre-allocate tensors
        if self.optimized and self.task_type == "multi-label":
            self.targets = None
            self.anti_targets = None
            self.xs_pos = None
            self.xs_neg = None
            self.asymmetric_w = None
            self.loss = None

    def forward(self, inputs, targets, **kwargs):
        """
        Forward pass to compute the Asymmetric Loss.
        
        Args:
            inputs: Predictions (logits) from the model.
            targets: Ground truth labels.
            **kwargs: Additional keyword arguments (for compatibility with training framework).
        
        Returns:
            Loss tensor with the reduction option applied.
        """
        if self.task_type == "multi-label":
            if self.optimized:
                return self._multi_label_asymmetric_loss_optimized(inputs, targets)
            else:
                return self._multi_label_asymmetric_loss(inputs, targets)
        elif self.task_type == "single-label":
            return self._single_label_asymmetric_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'multi-label' or 'single-label'."
            )

    def _multi_label_asymmetric_loss(self, x, y):
        """
        Standard asymmetric loss for multi-label classification.
        
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            grad_ctx = (
                torch.no_grad() if self.disable_torch_grad_focal_loss else nullcontext()
            )
            with grad_ctx:
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _multi_label_asymmetric_loss_optimized(self, x, y):
        """
        Optimized version - minimizes memory allocation and gpu uploading,
        favors inplace operations.
        
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            grad_ctx = (
                torch.no_grad() if self.disable_torch_grad_focal_loss else nullcontext()
            )
            with grad_ctx:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                              self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w

        loss = -self.loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _single_label_asymmetric_loss(self, inputs, target):
        """
        Asymmetric loss for single-label classification problems.
        
        "input" dimensions: - (batch_size, number_classes)
        "target" dimensions: - (batch_size)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        grad_ctx = (
                torch.no_grad() if self.disable_torch_grad_focal_loss else nullcontext()
            )
        with grad_ctx:
            xs_pos = xs_pos * targets
            xs_neg = xs_neg * anti_targets
            asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
            log_preds = log_preds * asymmetric_w

            # loss calculation
            loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
