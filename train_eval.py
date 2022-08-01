import numpy as np

import torch
from loss import GDROLoss


def train(dataloader, model, loss_fn, optimizer, verbose=False):
    """
    Train the model for one epoch
    :param dataloader: The dataloader for the training data
    :param model: The model to train
    :param loss_fn: The loss function to use for training
    :param optimizer: The optimizer to use for training
    :param verbose: Whether to print the average training loss of the epoch
    :return:
    """
    model.train()

    steps_per_epoch = dataloader.batches_per_epoch()

    avg_loss = 0

    for i in range(steps_per_epoch):
        loss = loss_fn(next(dataloader))
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss /= steps_per_epoch

    if verbose:
        print("Average training loss:", avg_loss)


def evaluate(dataloader, model, num_subclasses, vector_subclass=False, verbose=False):
    """
    Evaluate the model's accuracy and subclass sensitivities
    :param dataloader: The dataloader for the validation/testing data
    :param model: The model to evaluate
    :param num_subclasses: The number of subclasses to evaluate on, this should be equal to the number of subclasses present in the data
    :param verbose: Whether to print the results
    :return: A tuple containing the overall accuracy and the sensitivity for each subclass
    """
    model.eval()

    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    with torch.no_grad():
        X = dataloader.dataset.features
        y = dataloader.dataset.labels
        c = dataloader.dataset.subclasses

        pred = model(X)

        for subclass in range(num_subclasses):
            if vector_subclass:
                subclass_idx = c[:,subclass] == 1
            else:
                subclass_idx = c == subclass

            num_samples[subclass] += torch.sum(subclass_idx)
            subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(
                torch.float).sum().item()

    subgroup_accuracy = subgroup_correct / num_samples

    
    # for calculating overall accuracy, we simply recalculate
    # as the subclasses may not be a complete partition of data
    accuracy = (pred.argmax(1) == y).type(
                torch.float).sum().item()/ len(y)

    if verbose:
        print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:",
              min(subgroup_accuracy))

    return (accuracy, *subgroup_accuracy)


def train_epochs(epochs,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler=None,
                 vector_subclass=False,
                 verbose=False,
                 record=False,
                 num_subclasses=1):
    """
    Trains the model for a number of epochs and evaluates the model at each epoch
    :param epochs: The number of epochs to train
    :param train_dataloader: The dataloader for the training data
    :param test_dataloader: The dataloader for the validation/testing data
    :param model: The model to train and evaluate
    :param loss_fn: The loss function to use for training
    :param optimizer: The optimizer to use for training
    :param scheduler: The learning rate scheduler, if any, to use for training
    :param verbose: Whether to print the epoch number and the results for each epoch
    :param record: Whether to return the results from evaluate, generally this should be True
    :param num_subclasses: The number of subclasses to evaluate on
    :return: A list containing the overall accuracy and subclass sensitivities for each epoch, arranged 1-dimensionally ex. [accuracy_1, subclass1_1, subclass2_1, accuracy_2, subclass1_2, subclass2_2...]
    """
    if record:
        accuracies = []
        q_data = None
        if isinstance(loss_fn, GDROLoss):
            q_data = []

    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')

        train(train_dataloader, model, loss_fn, optimizer)
        if scheduler:
            scheduler.step(evaluate(test_dataloader, model, num_subclasses=num_subclasses)[0])

        if record:
            epoch_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses, vector_subclass=vector_subclass)
            accuracies.extend(epoch_accuracies)
            if isinstance(loss_fn, GDROLoss):
                q_data.extend(loss_fn.q.tolist())

    if record:
        return accuracies, q_data
    else:
        return None


def run_trials(num_trials,
               epochs,
               train_dataloader,
               test_dataloader,
               model_class,
               model_args,
               loss_class,
               loss_args,
               optimizer_class,
               optimizer_args,
               device='cpu',
               num_subclasses=1,
               scheduler_class=None,
               scheduler_args=None,
               verbose=False,
               record=False):
    """
    Runs a number of trials
    :param num_trials: The number of trials to run
    :param epochs: The number of epochs to train per trial
    :param train_dataloader: The dataloader for the training data
    :param test_dataloader: The dataloader for the validation/test data
    :param model_class: Class of the model to train and evaluate. Must be a class so it can be initialized at the beginning of each trial
    :param model_args: kwargs to input for the model class constructor
    :param loss_class: Class of the loss function to use for training
    :param loss_args: kwargs to input for the loss class constructor
    :param optimizer_class: Class of the optimizer to use for training
    :param optimizer_args: kwargs to input for the optimizer class constructor
    :param device: The device to use (either cpu or cuda)
    :param num_subclasses: The number of subclasses to evaluate
    :param scheduler_class: The class of the learning rate scheduler, if any, to use
    :param scheduler_args: kwargs for the scheduler
    :param verbose: Whether to print trial number as well as epoch number and results per epoch
    :param record: Whether to record the results, this should almost always be True
    :return: A list containing the results for each epoch for each trial, again arranged 1-dimensionally
    """
    if scheduler_args is None:
        scheduler_args = {}
    if record:
        accuracies = []
        roc_data = [None, None]
        q_data = None
        if loss_class is GDROLoss:
            q_data = []

    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        model = model_class(**model_args).to(device)
        loss_args['model'] = model
        loss_fn = loss_class(**loss_args)
        optimizer_args['params'] = model.parameters
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        scheduler_args['optimizer'] = optimizer
        scheduler = scheduler_class(**scheduler_args)

        trial_results = train_epochs(epochs,
                                     train_dataloader,
                                     test_dataloader,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     scheduler,
                                     verbose,
                                     record,
                                     num_subclasses)

        if record:
            trial_accuracies, trial_q_data, trial_g_data = trial_results
            accuracies.extend(trial_accuracies)

            if isinstance(loss_fn, GDROLoss):
                q_data.extend(trial_q_data)

            with torch.no_grad():
                preds = model(test_dataloader.dataset.features)
                probabilities = torch.nn.functional.softmax(preds, dim=1)[:, 1]
                if roc_data[0] is None:
                    roc_data[0] = probabilities
                else:
                    roc_data[0] += probabilities
    if record:
        roc_data[0] /= num_trials
        labels = test_dataloader.dataset.labels
        roc_data[1] = labels

    if record:
        return accuracies, q_data, roc_data
    else:
        return None
