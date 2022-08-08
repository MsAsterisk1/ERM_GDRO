import torch


class ERMLoss:
    """
    Implements a standard Empirical Risk Minimization loss function
    Takes the classifier model and an underlying loss function as input, ex. a neural network for the model and cross-entropy loss as the underlying function
    """

    def __init__(self, model, loss_fn):
        self.accumulated = torch.tensor(0.0)
        self.model = model
        self.loss_fn = loss_fn

    def compute_loss(self, preds, y, accumulate=False):
        if self.accumulated.device != y.device:
            self.accumulated = self.accumulated.to(y.device)
        self.accumulated += self.loss_fn(preds, y)

        loss = self.accumulated

        if not accumulate:
            self.accumulated = torch.tensor(0.0)

        return loss

    def __call__(self, minibatch, accumulate=False):
        X, y, _ = minibatch

        return self.compute_loss(self.model(X), y, accumulate)


class GDROLoss:
    """
    Implements the gDRO loss function
    See https://arxiv.org/abs/1911.08731 for details on the algorithm
    """

    def __init__(self, model, loss_fn, eta, num_subclasses, vector_subclass=False):
        self.accumulated = [torch.zeros(num_subclasses), torch.zeros(num_subclasses)]
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.ones(num_subclasses) / num_subclasses
        self.eta = eta
        self.vector_subclass = vector_subclass
        self.num_subclasses = num_subclasses

    def compute_loss(self, preds, y, c, accumulate=False):
        device = y.device
        self.accumulated[0] = self.accumulated[0].to(device)
        self.accumulated[1] = self.accumulated[1].to(device)

        if self.q.device != device:
            self.q = self.q.to(device)

        losses = self.accumulated[0]
        subclass_counts = self.accumulated[1]

        for subclass in range(self.num_subclasses):
            if self.vector_subclass:
                subclass_idx = c[:,subclass] == 1
            else:
                subclass_idx = c == subclass

            subclass_counts[subclass] += torch.sum(subclass_idx)

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] += self.loss_fn(preds[subclass_idx], y[subclass_idx])

        self.accumulated = [losses, subclass_counts]

        # update q
        if self.model.training and not accumulate:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        if not accumulate:
            self.accumulated = [torch.zeros(self.num_subclasses), torch.zeros(self.num_subclasses)]

        return loss

    def __call__(self, minibatch, accumulate=False):
        
        X, y, c = minibatch

        preds = self.model(X)

        return self.compute_loss(preds, y, c, accumulate)


class ERMGDROLoss:
    """
    Combines ERM and gDRO loss functions using t as the interpolation parameter
    Calculates the loss as (t*ERM + (1-t)*gDRO)
    While this class does not have any method to change t internally, the value can be set from outside to produce more dynamic behavior
    """
    def __init__(self, model, loss_fn, eta, num_subclasses, t=1, vector_subclass=False):
        self.erm = ERMLoss(model, loss_fn)
        self.gdro = GDROLoss(model, loss_fn, eta, num_subclasses, vector_subclass)
        self.model = model
        self.t = t

    def compute_loss(self, preds, y, c, accumulate=False):
        return self.t * self.erm.compute_loss(preds, y, accumulate) + \
               (1 - self.t) * self.gdro.compute_loss(preds, y, c, accumulate)

    def __call__(self, minibatch, accumulate=False):
        X, y, c = minibatch
        preds = self.model(X)
        return self.compute_loss(preds, y, c, accumulate)


class UpweightLoss:
    """
    Loss function which takes a weighted average of the losses over each subclasses, weighted by the inverse of the size of the subclass
    In other words, UpweightLoss upweights the loss of small subclasses as compared to larger ones.
    Equivalent to gDRO with eta=0
    """
    def __init__(self, model, loss_fn, num_subclasses):
        # wraps a GDROLoss with eta=0
        self.gdro = GDROLoss(model, loss_fn, eta=0, num_subclasses=num_subclasses)

    def compute_loss(self, preds, y, c, accumulate=False):
        return self.gdro.compute_loss(preds, y, c, accumulate)

    def __call__(self, minibatch, accumulate=False):
        return self.gdro(minibatch, accumulate)
