import numpy as np

import torch
from loss import GDROLoss


def train(dataloader, model, loss_fn, optimizer, verbose=False, sub_batches=1, scheduler=None, gradient_clip=None):
    """
    Train the model for one epoch
    :param dataloader: The dataloader for the training data
    :param model: The model to train
    :param loss_fn: The loss function to use for training
    :param optimizer: The optimizer to use for training
    :param verbose: Whether to print the average training loss of the epoch
    :param sub_batches: Number of sub-batches to split a batch into, useful when the full batch can't fit in memory
    """
    model.train()

    steps_per_epoch = dataloader.batches_per_epoch()

    avg_loss = 0

    for i in range(steps_per_epoch):
        for s in range(sub_batches):
            # accumulate is True except on the last sub-batch
            accumulate = s != sub_batches - 1
            loss = loss_fn(next(dataloader), accumulate=accumulate)
        loss /= sub_batches
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    avg_loss /= steps_per_epoch

    if verbose:
        print("Average training loss:", avg_loss)


def evaluate(dataloader, model, num_subclasses, vector_subclass=False, replacement=False, get_loss=False, verbose=False):
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

        if replacement: #if dataloader samples with replacement, can only use dataset
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
            accuracy = (pred.argmax(1) == y).type(
            torch.float).sum().item()/ len(y)

        else: #if dataloader does not replace, can use batches
            steps_per_epoch = dataloader.batches_per_epoch()
            accuracy = 0
            if get_loss:
                loss = 0
                loss_fn = torch.nn.CrossEntropyLoss()

            for i in range(steps_per_epoch):
                minibatch = next(dataloader)
                X,y,c = minibatch

                pred = model(X)

                for subclass in range(num_subclasses):
                    if vector_subclass:
                        subclass_idx = c[:,subclass] == 1
                    else:
                        subclass_idx = c == subclass                    
                
                    num_samples[subclass] += torch.sum(subclass_idx)


                    if torch.sum(subclass_idx) > 0:
                        subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(torch.float).sum().item()
                
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
                if get_loss:
                    #accumulate loss over entire epoch
                    loss += loss_fn(pred, y)
            
            if get_loss:
                loss /= steps_per_epoch
            subgroup_accuracy = subgroup_correct / num_samples

            accuracy /= len(dataloader.dataset)


    if verbose:
        if get_loss:
            print('Loss:', loss.item(), "Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:",
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
                 optimizer,
                 scheduler=None,
                 vector_subclass=False,
                 verbose=False,
                 record=False,
                 save_weights_name=None,
                 num_subclasses=1,
                 gradient_clip = None,
                 sub_batches=1):
    """
    Trains the model for a number of epochs and evaluates the model at each epoch
    :param epochs: The number of epochs to train
    :param train_dataloader: The dataloader for the training data
    :param val_dataloader: The dataloader for the validation data
    :param test_dataloader: The dataloader for the testing data
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
        accuracies = list(evaluate(test_dataloader, model, num_subclasses=num_subclasses, vector_subclass=vector_subclass, verbose=verbose))
        q_data = None
        if isinstance(loss_fn, GDROLoss):
            q_data = loss_fn.q.tolist()

    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')

        train(train_dataloader, model, loss_fn, optimizer, verbose=verbose, sub_batches=sub_batches, scheduler=scheduler, gradient_clip=gradient_clip)
        # if scheduler:
        #     scheduler.step(evaluate(val_dataloader, model, num_subclasses=num_subclasses)[0])

        if record:
            epoch_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses, vector_subclass=vector_subclass, verbose=verbose)
            accuracies.extend(epoch_accuracies)
            if isinstance(loss_fn, GDROLoss):
                q_data.extend(loss_fn.q.tolist())

        if save_weights_name is not None:
            print(f'For Cross Val:')
            _ = evaluate(val_dataloader, model, num_subclasses, vector_subclass=vector_subclass, get_loss=True, verbose=True)
            torch.save(model.state_dict(), f'./epoch_{epoch+1}_{save_weights_name}.wt')

    if record:
        return accuracies, q_data
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
               device='cpu',
               num_subclasses=1,
               scheduler_class=None,
               scheduler_args=None,
               verbose=False,
               record=False,
               sub_batches=1):
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
        optimizer_args['params'] = model.parameters()
        optimizer = optimizer_class(**optimizer_args)
        if scheduler_class is not None:
            scheduler_args['optimizer'] = optimizer
            scheduler = scheduler_class(**scheduler_args)
        else:
            scheduler = None

        trial_results = train_epochs(epochs=epochs,
                                     train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader,
                                     test_dataloader=test_dataloader,
                                     model=model,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     verbose=verbose,
                                     record=record,
                                     num_subclasses=num_subclasses,
                                     sub_batches=sub_batches
                                     )

        if record:
            trial_accuracies, trial_q_data = trial_results
            accuracies.extend(trial_accuracies)

            if isinstance(loss_fn, GDROLoss):
                q_data.extend(trial_q_data)

            # with torch.no_grad():
                # X = test_dataloader.dataset.features
                # if len(X) == 1:
                #     X = X[0]
                # preds = model(X)
                # probabilities = torch.nn.functional.softmax(preds, dim=1)[:, 1]
                # if roc_data[0] is None:
                #     roc_data[0] = probabilities
                # else:
                #     roc_data[0] += probabilities
    # if record:
        # roc_data[0] /= num_trials
        # labels = test_dataloader.dataset.labels
        # roc_data[1] = labels

    if record:
        return accuracies, q_data, roc_data
    else:
        return None
