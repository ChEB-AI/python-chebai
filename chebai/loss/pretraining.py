import torch


class ElectraPreLoss(torch.nn.Module):
    """
    Custom loss module for pre-training ELECTRA-like models.

    This module computes a combined loss from two CrossEntropyLosses:
    one for generator predictions and another for discriminator predictions.

    Attributes:
        ce (torch.nn.CrossEntropyLoss): Cross entropy loss function.

    Methods:
        forward(input, target, **loss_kwargs):
            Computes the combined loss for generator and discriminator predictions.

    """

    def __init__(self):
        """
        Initializes the ElectraPreLoss module.
        """
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target, **loss_kwargs):
        """
        Forward pass for computing the combined loss.

        Args:
            input (tuple): A tuple containing generator predictions (gen_pred, disc_pred).
            target (tuple): A tuple containing generator targets (gen_tar, disc_tar).
            **loss_kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Combined loss of generator and discriminator predictions.
        """
        t, p = input
        gen_pred, disc_pred = t
        gen_tar, disc_tar = p

        # Compute losses for generator and discriminator
        gen_loss = self.ce(target=torch.argmax(gen_tar.int(), dim=-1), input=gen_pred)
        disc_loss = self.ce(
            target=torch.argmax(disc_tar.int(), dim=-1), input=disc_pred
        )
        return gen_loss + disc_loss
