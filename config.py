# train neighborsampling config
hyperparameter_defaults = dict(
    device=1, 
    log_steps = 5, 
    model_name = "sage", 
    num_layers = 3, 
    hidden_channels = 256, 
    lr = 0.01, 
    epochs = 300, 
    runs = 1, 
    inference_mode = "batch",
    wandb = True,
    use_virtual = True, # use virtual node data
    num_parts = 100,
    small_trainingset=1.0, # 1.0 means all train dataset

    neighbor1=6, # 15, 10, 5
    neighbor2=5,
    neighbor3=5,
    dropout=0.5,
    batch_size=1024,

    # wandb
    experiment_name="virtual_neighborsampling"

)

sweep_config = {
    "name": "baseline_neighborsampling",
    "metric": {"name": "val/acc", "goal": "maximize"},
    'method': 'grid',
    "parameters":{
        "neighbor1": {
            "values": [2, 4, 6],
            },
        "neighbor2": {
            "values": [2, 4, 6],
            },
        "neighbor3": {
            "values": [2, 4, 6],
            },
        "batch_size": {
            "values": [1024, 64],
        },
        "dropout":  {
            "values": [0.5],
        },
    }
}