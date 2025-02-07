import numpy as np

import torch
from torch import nn
from loss import CRISLoss
from tqdm import tqdm


def train(dataloader, model, loss_fn, optimizer, verbose=False, scheduler=None, gradient_clip=None, use_tqdm=False):
    """
    Train the model for one epoch
    :param dataloader: The dataloader for the training data
    :param model: The model to train
    :param loss_fn: The loss function to use for training
    :param optimizer: The optimizer to use for training
    :param verbose: Whether to print the average training loss of the epoch
    :param scheduler: Learning rate scheduler to use
    :param gradient_clip: Gradient clipping to use
    :param use_tqdm: Whether to use tqdm progress bar when iterating through the data
    """
    model.train()

    steps_per_epoch = dataloader.batches_per_epoch()

    avg_loss = 0

    step_iter = tqdm(range(steps_per_epoch)) if use_tqdm else range(steps_per_epoch)

    for _ in step_iter:
        loss = loss_fn(next(dataloader))
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    avg_loss /= steps_per_epoch

    if verbose:
        print("Average training loss:", avg_loss)


def evaluate(dataloader, model, num_subclasses, vector_subclass=False, get_loss=False, verbose=False, subclass_labels=False):
    """
    Evaluate the model's accuracy and subclass sensitivities
    :param dataloader: The dataloader for the validation/testing data
    :param model: The model to evaluate
    :param num_subclasses: The number of subclasses to evaluate on, this should be equal to the number of subclasses present in the data
    :param vector_subclass: True if the subclass is represented by more than one number (such as CivilComments)
    :param get_loss: Calculate the average cross-entropy loss as well as the accuracy and subclass sensitivities
    :param verbose: Whether to print the results
    :param subclass_labels: Whether to evaluate on the subclass labels rather than the superclass labels
    :return: A tuple containing the overall accuracy and the sensitivity for each subclass
    """
    model.eval()

    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    with torch.no_grad():
        steps_per_epoch = dataloader.batches_per_epoch()
        accuracy = 0
        if get_loss:
            loss = 0
            loss_fn = torch.nn.CrossEntropyLoss()

        for i in range(steps_per_epoch):
            minibatch = next(dataloader)
            X, y, c = minibatch

            pred = model(X)

            for subclass in range(num_subclasses):
                if vector_subclass:
                    subclass_idx = c[:, subclass] == 1
                else:
                    subclass_idx = c == subclass

                num_samples[subclass] += torch.sum(subclass_idx)

                if torch.sum(subclass_idx) > 0:
                    if subclass_labels:
                        # Assumes that the first half of the subtypes is class 0 and the second half is class 1
                        # Also assumes that unlike during training, y refers to the superclass labels
                        subgroup_correct[subclass] += (
                                    (pred[subclass_idx].argmax(1) >= num_subclasses // 2) == (y[subclass_idx] >= num_subclasses // 2)).type(
                            torch.float).sum().item()
                    else:
                        subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(
                            torch.float).sum().item()

            if subclass_labels:
                accuracy += ((pred.argmax(1) >= num_subclasses // 2) == (y >= num_subclasses // 2)).type(torch.float).sum().item()
            else:
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
            if get_loss:
                # accumulate loss over entire epoch
                loss += loss_fn(pred, c if subclass_labels else y)

        if get_loss:
            loss /= steps_per_epoch
        subgroup_accuracy = subgroup_correct / num_samples

        accuracy /= len(dataloader.dataset)

    if verbose:
        if get_loss:
            print('Loss:', loss.item(), "Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy,
                  "\nWorst Group Accuracy:",
                  min(subgroup_accuracy))
        else:
            print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:",
                  min(subgroup_accuracy))
    if get_loss:
        return (loss, accuracy, *subgroup_accuracy)

    return (accuracy, *subgroup_accuracy)


def train_epochs(epochs,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 model,
                 loss_fn,
                 optimizer_class,
                 optimizer_args,
                 scheduler_class=None,
                 scheduler_args=None,
                 vector_subclass=False,
                 verbose=False,
                 record=False,
                 validation=None,
                 num_subclasses=1,
                 gradient_clip=None,
                 subclass_labels=False,
                 use_tqdm=False):
    """
    Trains the model for a number of epochs and evaluates the model at each epoch
    :param epochs: The number of epochs to train
    :param train_dataloader: The dataloader for the training data
    :param val_dataloader: The dataloader for the validation data
    :param test_dataloader: The dataloader for the testing data
    :param model: The model to train and evaluate
    :param loss_fn: The loss function to use for training
    :param optimizer_class: Class of the optimizer to use for training
    :param optimizer_args: kwargs to input for the optimizer class constructor
    :param scheduler_class: The class of the learning rate scheduler, if any, to use
    :param scheduler_args: kwargs for the scheduler
    :param vector_subclass: True if the subclass is represented by more than one number (such as CivilComments)
    :param verbose: Whether to print the epoch number and the results for each epoch
    :param record: Whether to return the results from evaluate, generally this should be True
    :param validation: If equal to 0, best overall validation accuracy is tracked and printed at the end. If another integer, best worst-group accuracy is used instead. If None, validation accuracy is not tracked
    :param num_subclasses: The number of subclasses to evaluate on
    :param gradient_clip: Gradient clipping to use
    :param subclass_labels: Whether to evaluate on the subclass labels rather than the superclass labels
    :param use_tqdm: Whether to use tqdm progress bar
    :return: A list containing the overall accuracy and subclass sensitivities for each epoch, arranged 1-dimensionally ex. [accuracy_1, subclass1_1, subclass2_1, accuracy_2, subclass1_2, subclass2_2...]
    """
    optimizer = optimizer_class(**optimizer_args)
    if scheduler_class is not None:
        scheduler_args['optimizer'] = optimizer
        scheduler = scheduler_class(**scheduler_args)
    else:
        scheduler = None

    if record:
        accuracies = list(
            evaluate(test_dataloader, model, num_subclasses=num_subclasses, vector_subclass=vector_subclass,
                     verbose=verbose, subclass_labels=subclass_labels))

    if isinstance(loss_fn, CRISLoss):
        epochs *= 2
        validation = 0

    if validation is not None:
        v = evaluate(val_dataloader, model, vector_subclass=vector_subclass, num_subclasses=num_subclasses, verbose=verbose, subclass_labels=subclass_labels)[1:]

        # First validation measurement is best so far
        best_model = model.state_dict().copy()
        # 0: use overall accuracy to evaluate model
        # not 0: use worst sensitivity to evaluate model
        best_val = min(v) if validation else (sum(v) / len(v))
        best_epoch = 0

    for epoch in tqdm(range(epochs)):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')

        if isinstance(loss_fn, CRISLoss) and (epoch == epochs // 2):
            print("CRIS switching to gDRO")
            # Set CRIS to use gDRO
            loss_fn.erm_mode = False

            # Load best model weights, replace last layer with 2 outputs
            model.load_state_dict(best_model)
            model.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=2, bias=True, device=model.device),
            )

            # Reinitialize optimizer and optional scheduler
            optimizer_args['params'] = model.parameters()
            optimizer = optimizer_class(**optimizer_args)
            if scheduler_class is not None:
                scheduler_args['optimizer'] = optimizer
                scheduler = scheduler_class(**scheduler_args)
            else:
                scheduler = None

            # Freeze featurizer
            model.set_grad('featurizer', False)

            # Once CRIS switches to gDRO we want it to train for the binary classification task
            # Train dataloader is partitioned and we need to change the dataset for gdro, so dataloader1
            # Dataloader1 is a regular loader so it has a dataset, but that dataset is a subdataset so it in turn
            # has a dataset
            # That dataset finally has the subclass_labels attribute, which needs to be set to False
            train_dataloader.dataloader1.dataset.dataset.subclass_labels = False

            # Reset running best validation accuracy and switch to using worst-group sensitivity
            best_val = 0
            validation = 1

        train(train_dataloader, model, loss_fn, optimizer, verbose=verbose,
              scheduler=scheduler, gradient_clip=gradient_clip, use_tqdm=use_tqdm)

        if validation is not None:
            v = evaluate(val_dataloader, model, vector_subclass=vector_subclass, num_subclasses=num_subclasses,
                         verbose=verbose, subclass_labels=subclass_labels)[1:]

            if best_val < (min(v) if validation else (sum(v) / len(v))):
                best_model = model.state_dict().copy()
                best_val = min(v) if validation else (sum(v) / len(v))
                best_epoch = epoch + 1

        if record:
            epoch_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses,
                                        vector_subclass=vector_subclass, verbose=verbose, subclass_labels=subclass_labels)
            accuracies.extend(epoch_accuracies)

    if validation is not None:
        print(f"Best epoch using cross-validation: {best_epoch}")

    if record:
        return accuracies
    else:
        return None


def run_trials(num_trials,
               epochs,
               train_dataloader,
               val_dataloader,
               test_dataloader,
               model_class,
               model_args,
               loss_class,
               loss_args,
               optimizer_class,
               optimizer_args,
               num_subclasses=1,
               scheduler_class=None,
               scheduler_args=None,
               verbose=False,
               record=False,
               validation=None,
               gradient_clip=None,
               vector_subclass=False,
               subclass_labels=False):
    """
    Runs a number of trials
    :param num_trials: The number of trials to run
    :param epochs: The number of epochs to train per trial
    :param train_dataloader: The dataloader for the training data
    :param val_dataloader: The dataloader for the validation data
    :param test_dataloader: The dataloader for the test data
    :param model_class: Class of the model to train and evaluate. Must be a class so it can be initialized at the beginning of each trial
    :param model_args: kwargs to input for the model class constructor
    :param loss_class: Class of the loss function to use for training
    :param loss_args: kwargs to input for the loss class constructor
    :param optimizer_class: Class of the optimizer to use for training
    :param optimizer_args: kwargs to input for the optimizer class constructor
    :param num_subclasses: The number of subclasses to evaluate
    :param scheduler_class: The class of the learning rate scheduler, if any, to use
    :param scheduler_args: kwargs for the scheduler
    :param verbose: Whether to print trial number as well as epoch number and results per epoch
    :param record: Whether to record the results, this should almost always be True
    :param validation: If equal to 0, best overall validation accuracy is tracked and printed at the end of each trial. If another integer, best worst-group accuracy is used instead. If None, validation accuracy is not tracked
    :param gradient_clip Gradient clipping to use
    :param vector_subclass True if the subclass is represented by more than one number (such as CivilComments)
    :param subclass_labels: Whether to evaluate on the subclass labels rather than the superclass labels
    :return: A list containing the results for each epoch for each trial, arranged 1-dimensionally
    """
    if scheduler_args is None:
        scheduler_args = {}
    if record:
        accuracies = []

    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        model = model_class(**model_args)
        loss_args['model'] = model
        loss_fn = loss_class(**loss_args)
        optimizer_args['params'] = model.parameters()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trial_results = train_epochs(epochs=epochs,
                                     train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader,
                                     test_dataloader=test_dataloader,
                                     model=model,
                                     loss_fn=loss_fn,
                                     optimizer_class=optimizer_class,
                                     optimizer_args=optimizer_args,
                                     scheduler_class=scheduler_class,
                                     scheduler_args=scheduler_args,
                                     verbose=verbose,
                                     record=record,
                                     validation=validation,
                                     num_subclasses=num_subclasses,
                                     gradient_clip=gradient_clip,
                                     vector_subclass=vector_subclass,
                                     subclass_labels=subclass_labels
                                     )

        if record:
            trial_accuracies = trial_results
            accuracies.extend(trial_accuracies)

    if record:
        return accuracies
    else:
        return None
