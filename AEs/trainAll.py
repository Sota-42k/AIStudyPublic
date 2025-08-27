from simpleAE import ae_training
from simpleVAE import vae_training

from mutualConnectionAE import mutual_train as ae_mutual_train
from mutualConnectionVAE import mutual_train as vae_mutual_train

def trainAll():
    print("Training Simple AE...")
    ae_training(epochs=10, save=True)
    print("Training Simple VAE...")
    vae_training(epochs=10, save=True)
    print("Training Mutual Connection AE...")
    ae_mutual_train(loops=10, save=True)
    print("Training Mutual Connection VAE...")
    vae_mutual_train(loops=10, save=True)