import torch

class ElectraPreLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target, **loss_kwargs):
        t, p = input
        gen_pred, disc_pred = t
        gen_tar, disc_tar = p
        gen_loss = self.ce(target=torch.argmax(gen_tar.int(), dim=-1), input=gen_pred)
        disc_loss = self.ce(
            target=torch.argmax(disc_tar.int(), dim=-1), input=disc_pred
        )
        return gen_loss + disc_loss

