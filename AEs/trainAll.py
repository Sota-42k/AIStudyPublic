
from simpleAE import ae_train
from simpleVAE import vae_train
from mutualConnectionAE import mutual_train as ae_mutual_train
from mutualConnectionVAE import mutual_train as vae_mutual_train
from randomConnectionAE import random_train as ae_random_train
from randomConnectionVAE import random_train as vae_random_train

def trainAll():
    print("Training Simple AE...")
    ae_train(
        epochs=300,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )
    print("Training Simple VAE...")
    vae_train(
        epochs=300,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )
    print("Training Mutual Connection AE...")
    ae_mutual_train(
        pretrain_epochs=300,
        loops=50,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )
    print("Training Mutual Connection VAE...")
    vae_mutual_train(
        pretrain_epochs=300,
        loops=50,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )
    print("Training Random Connection AE...")
    ae_random_train(
        num_aes=16,
        pretrain_epochs=50,
        loops=50,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )
    print("Training Random Connection VAE...")
    vae_random_train(
        num_vaes=16,
        pretrain_epochs=50,
        loops=50,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

if __name__ == "__main__":
    trainAll()