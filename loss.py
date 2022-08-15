import torch


class ERMLoss:
    """
    Implements a standard Empirical Risk Minimization loss function
    Takes the classifier model and an underlying loss function as input, ex. a neural network for the model and cross-entropy loss as the underlying function
    """

    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def compute_loss(self, preds, y):
        return self.loss_fn(preds, y)

    def __call__(self, minibatch):
        X, y, _ = minibatch

        return self.compute_loss(self.model(X), y)


class GDROLoss:
    """
    Implements the gDRO loss function
    See https://arxiv.org/abs/1911.08731 for details on the algorithm
    """

    def __init__(self, model, loss_fn, eta, num_subclasses):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.ones(num_subclasses) / num_subclasses
        self.eta = eta
        self.num_subclasses = num_subclasses

    def compute_loss(self, preds, y, c):
        device = y.device

        if self.q.device != device:
            self.q = self.q.to(device)

        losses = torch.zeros(self.num_subclasses).to(device)
        subclass_counts = torch.zeros(self.num_subclasses).to(device)

        for subclass in range(self.num_subclasses):

            subclass_idx = c == subclass

            subclass_counts[subclass] += torch.sum(subclass_idx)

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] += self.loss_fn(preds[subclass_idx], y[subclass_idx])

        # update q
        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        return loss

    def __call__(self, minibatch):
        
        X, y, c = minibatch

        preds = self.model(X)

        return self.compute_loss(preds, y, c)


class ERMGDROLoss:
    """
    Combines ERM and gDRO loss functions using t as the interpolation parameter
    Calculates the loss as (t*ERM + (1-t)*gDRO)
    While this class does not have any method to change t internally, the value can be set from outside to produce more dynamic behavior
    """
    def __init__(self, model, loss_fn, eta, num_subclasses, t=1, partitioned=False, prop=False):
        self.erm = ERMLoss(model, loss_fn)
        self.gdro = GDROLoss(model, loss_fn, eta, num_subclasses)
        self.model = model
        self.t = t
        self.partitioned = partitioned
        self.prop = prop

    def compute_loss(self, preds, y, c):
        
        if self.partitioned:
            return self.t * self.erm.compute_loss(preds[0], y[0]) + (1 - self.t) * self.gdro.compute_loss(preds[1], y[1], c[1])
        else:  
            return self.t * self.erm.compute_loss(preds, y) + (1 - self.t) * self.gdro.compute_loss(preds, y, c)

    def __call__(self, minibatch):
        X, y, c = minibatch

        if self.partitioned:
            if self.prop:
                self.model.toggle_featurizer()
                pred_gdro = self.model(X[1])
                
                self.model.toggle_featurizer()
                pred_erm = self.model(X[0])

                preds = (pred_erm, pred_gdro)
            else:
                preds = (self.model(X[0]), self.model(X[1]))
        else:
            preds = self.model(X)

        return self.compute_loss(preds, y, c)


class CRISLoss:
    def __init__(self, model, loss_fn, eta, num_subclasses):
        self.erm = ERMLoss(model, loss_fn)
        self.gdro = GDROLoss(model, loss_fn, eta, num_subclasses)
        self.model = model
        self.erm_mode = True

    def compute_loss(self, preds, y, c):
        if self.erm_mode:
            return self.erm.compute_loss(preds, y)
        else:
            return self.gdro.compute_loss(preds, y, c)

    def reset_gdro(self):
        self.gdro = GDROLoss(self.gdro.model, self.gdro.loss_fn, self.gdro.eta, self.gdro.num_subclasses)

    def toggle_mode(self):
        self.erm_mode = not self.erm_mode
        self.model.set_grad('featurizer', self.erm_mode)

    def __call__(self, minibatch):
        X, y, c = minibatch

        preds = self.model(X)

        return self.compute_loss(preds, y, c)
