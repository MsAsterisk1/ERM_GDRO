    os.mkdir(results_dir)

verbose = args.verbose


subtypes = ["Overall"]
subtypes.extend(list(range(run_trials_args['num_subclasses'])))

erm_class = ERMLoss
erm_args = {'loss_fn': torch.nn.CrossEntropyLoss()}
gdro_class = GDROLoss
gdro_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses}
upweight_class = ERMLoss
upweight_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'num_subclasses': num_subclasses}
mix25_class = ERMGDROLoss
mix25_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.25}
mix50_class = ERMGDROLoss
mix50_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.50}
mix75_class = ERMGDROLoss
mix75_args = {'loss_fn': torch.nn.CrossEntropyLoss(), 'eta': eta, 'num_subclasses': num_subclasses, 't': 0.75}

losses = {'erm': (erm_class, erm_args), 'gdro': (gdro_class, gdro_args), 'upweight': (upweight_class, upweight_args), 'mix25': (mix25_class, mix25_args), 'mix50': (mix50_class, mix50_args), 'mix75': (mix75_class, mix75_args)}

accuracies = {}

for loss_class, fn_name, loss_args in zip(
       # [erm_class, gdro_class, mix75_class, erm_class],
       # [erm_name,  gdro_name,  mix75_name,  upweight_name],
       # [erm_args,  gdro_args,  mix75_args,  erm_args]):
        # [erm_class, gdro_class, upweight_class, mix25_class, mix50_class, mix75_class],
        # [erm_name,  gdro_name,  upweight_name,  mix25_name,  mix50_name,  mix75_name],
        # [erm_args,  gdro_args,  upweight_args,  mix25_args,  mix50_args,  mix75_args]):torch.cuda.set_device(1)
        [mix50_class,],
        [mix50_name,],
        [mix50_args,]):

for loss in args.loss:
    if verbose:
        print(f"Running trials: {loss}")

    reweight_train = loss != 'erm'
    train_dataloader, val_dataloader, test_dataloader, = utils.get_dataloaders(
        (train_dataset, val_dataset, test_dataset),
        batch_size=batch_size,
        reweight_train=reweight_train
    )

    run_trials_args['loss_class'], run_trials_args['loss_args'] = losses[loss]

    run_trials_args['train_dataloader'] = train_dataloader
    run_trials_args['val_dataloader'] = val_dataloader
    run_trials_args['test_dataloader'] = test_dataloader

    accuracies[fn_name] = run_trials(**run_trials_args)[0]

    accuracies_df = pd.DataFrame(
        accuracies,
        index=pd.MultiIndex.from_product(
            [range(run_trials_args['num_trials']), range(run_trials_args['epochs'] + 1), subtypes],
            names=["trial", "epoch", "subtype"]
        )
    )

    accuracies_df.to_csv(results_dir + f'accuracies.csv')

