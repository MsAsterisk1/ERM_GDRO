import torch


class ERMLoss:
    """
    Implements a standard Empirical Risk Minimization loss function
    Takes the classifier model and an underlying loss function as input, ex. a neural network for the model and cross-entropy loss as the underlying function
    """

    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self, minibatch):
        # minibatch contains one batch of non-subtyped data
        X, y, _ = minibatch
        loss = self.loss_fn(self.model(X), y)

        return loss


class GDROLoss:
    """
    Implements the gDRO loss function
    See https://arxiv.org/abs/1911.08731 for details on the algorithm
    """

    def __init__(self, model, loss_fn, eta, num_subclasses, vector_subclass=False, normalize_loss=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.eta = eta
        self.vector_subclass=vector_subclass
        self.num_subclasses = num_subclasses
        self.normalize_loss = normalize_loss

    def __call__(self, minibatch):
        
        X,y,c = minibatch

        batch_size = y.shape[0]
        device = y.device

        if len(self.q) == 0:
            self.q = torch.ones(self.num_subclasses).to(device)
            self.q /= self.q.sum()

        losses = torch.zeros(self.num_subclasses).to(device)

        subclass_counts = torch.zeros(self.num_subclasses).to(device)

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass
        for subclass in range(self.num_subclasses):
            if self.vector_subclass:
                subclass_idx = c[:,subclass] == 1
            else:
                subclass_idx = c == subclass

            subclass_counts[subclass] = torch.sum(subclass_idx)

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        # update q
        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        if self.normalize_loss:
            losses *= subclass_counts
            loss = torch.dot(losses, self.q)
            loss /= batch_size
            loss *= self.num_subclasses
        else:
            loss = torch.dot(losses, self.q)

        return loss


class ERMGDROLoss:
    """
    Combines ERM and gDRO loss functions using t as the interpolation parameter
    Calculates the loss as (t*ERM + (1-t)*gDRO)
    While this class does not have any method to change t internally, the value can be set from outside to produce more dynamic behavior
    """
    def __init__(self, model, loss_fn, eta, num_subclasses, t):
        self.eta = eta
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.t = t
        self.num_subclasses = num_subclasses

    def __call__(self, minibatch):

        X, y, c = minibatch

        batch_size = X.shape[0]
        device = X.device

        if len(self.q) == 0:
            self.q = torch.ones(self.num_subclasses).to(device)
            self.q /= self.q.sum()

        losses = torch.zeros(self.num_subclasses).to(device)

        subclass_counts = torch.zeros(self.num_subclasses).to(device)

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass
        for subclass in range(self.num_subclasses):
            subclass_idx = c == subclass
            subclass_counts[subclass] = torch.sum(subclass_idx)

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        # update q
        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        # loss has to be normalized (subclasses scaled by their size) to be comparable to ERM (in which subclasses affect the loss in proportion to their size)
        losses *= subclass_counts
        gdro_loss = torch.dot(losses, self.q)
        gdro_loss /= batch_size
        gdro_loss *= self.num_subclasses

        erm_loss = torch.sum(losses)

        loss = self.t * erm_loss + (1 - self.t) * gdro_loss

        return loss, erm_loss, gdro_loss


class UpweightLoss:
    """
    Loss function which takes a weighted average of the losses over each subclasses, weighted by the inverse of the size of the subclass
    In other words, UpweightLoss upweights the loss of small subclasses as compared to larger ones.
    Equivalent to gDRO with eta=0 and normalize_loss=False
    """
    def __init__(self, model, loss_fn, num_subclasses):
        self.model = model
        self.loss_fn = loss_fn
        self.num_subclasses = num_subclasses

    def __call__(self, minibatch):
        X, y, c = minibatch

        device = X.device

        losses = torch.zeros(self.num_subclasses).to(device)

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass
        for subclass in range(self.num_subclasses):
            subclass_idx = c == subclass

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        loss = torch.sum(losses) / self.num_subclasses

        return loss
