
from timeit import default_timer as timer

import torch


class CoherenceLoss(torch.nn.Module):
    """
    Encourages spatial coherence for each part's similarity map.
    i.e. similarity is high in a single region of the image.
    Equivalent to the Dis Loss from ECCV 2018 Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
    by Ming Sun, et al.
    """
    def __init__(self, map_shape, device="cuda", reduce=False):
        super(CoherenceLoss, self).__init__()
        self.reduce = reduce
        self.map_shape = map_shape
        self.x = torch.arange(map_shape[1]).unsqueeze(0).to(device)
        self.y = torch.arange(map_shape[0]).unsqueeze(0).to(device)

    def forward(self, maps):
        """Forward pass for spatial coherence loss.

        Arguments
          maps (torch.Tensor): Attention maps of shape (batch_size, n_parts, map_height, map_width)
        Return
          loss (torch.Tensor): loss of shape (batch_size, n_parts) or (1,) if self.reduce
        """
        batch_size, n_parts, height, width = maps.shape

        max_val, max_ind = maps.view(batch_size, n_parts, -1).max(dim=-1)
        max_x = max_ind % width
        max_y = torch.div(max_ind, width, rounding_mode="trunc")
        x_distances = max_x.unsqueeze(-1) * self.x
        x_distances = x_distances ** 2
        y_distances = max_y.unsqueeze(-1) * self.y
        y_distances = y_distances ** 2

        # Use tensor broadcasting to get pairwise distances
        distances = x_distances.unsqueeze(-1) + y_distances.unsqueeze(-2)
        
        loss = maps * distances
        loss = loss.sum(dim=(2, 3))

        if self.reduce:
            loss = loss.mean()

        return loss


class NumProtoLoss(torch.nn.Module):
    """
    Encourage a maximum number of high contribution prototypes per image.
    """

    def __init__(self, n=4, reduce=False):
        super(NumProtoLoss, self).__init__()
        self.reduce = reduce
        self.n = n

    def forward(self, contributions):
        # Select indicies of top-n values
        n_samples, n_proto, n_class = contributions.shape
        args = contributions.argsort(dim=1, descending=True)[:, :self.n].unsqueeze(-1)
        mask = torch.ones_like(contributions)
        inds = torch.arange(n_samples).reshape((n_samples, 1, 1, 1)).repeat(1, args.shape[1], n_class, 1).to(contributions.device)
        other_inds = torch.arange(n_class).reshape((1, 1, n_class, 1)).repeat(n_samples, args.shape[1], 1, 1).to(contributions.device)
        zipped_args = torch.concat((inds, args, other_inds), dim=-1)
        mask[tuple(zipped_args.T)] = 0
        mask.requires_grad = True
        loss = contributions * mask
        if self.reduce:
            return loss.mean(dim=1)
        return loss


if __name__ == "__main__":
    protoLoss = NumProtoLoss()
    example_contributions = torch.rand((64, 100, 13))
    loss = protoLoss(example_contributions)
    loss.sum().backward()

    num = 10000
    cpus = torch.zeros(num)
    gpus = torch.zeros(num)
    torch.random.manual_seed(42)
    for i in range(num):
        example_maps = torch.rand((64, 100, 7, 7))
        example_maps.requires_grad = True
        coherence_loss_fn = CoherenceLoss((7, 7), device="cpu", reduce=False)
        t0 = timer()
        coherence_loss = coherence_loss_fn(example_maps)
        coherence_loss.mean().backward()
        t1 = timer()
        coherence_loss_fn = CoherenceLoss((7, 7), device="cuda", reduce=False)
        t2 = timer()
        coherence_loss = coherence_loss_fn(example_maps.cuda())
        coherence_loss.mean().backward()
        t3 = timer()
        cpus[i] = t1 - t0
        gpus[i] = t3 - t2
        del example_maps
        del coherence_loss_fn

    print(f"CPU: {cpus.mean()}")
    print(f"GPU: {gpus.mean()}")
    print("(batch, part):", coherence_loss)
    print("(part,):", coherence_loss.mean(dim=0))
    print(coherence_loss.mean())

    # Avg across 10,000 calculations

    # Using tensor broadcasting
    # (5, 10, 7, 7)
    #   CPU: 0.0001674366503721103
    #   GPU: 0.0002630125673022121
    # (64, 100, 7, 7)
    #   CPU: 0.0010432967683300376
    #   GPU: 0.0008729006513021886

    # Using a nested for loop
    # (5, 10, 7, 7)
    #   CPU: 0.0007591262692585588
    #   GPU: 0.001330850413069129
    # (64, 100, 7, 7)
    #   CPU: 0.0036183646880090237
    #   GPU: 0.0022741477005183697
