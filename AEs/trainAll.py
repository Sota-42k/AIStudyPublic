
from SimpleAE import ae_train
from SimpleVAE import vae_train
from MutualAE import mutual_train as ae_mutual_train
from MutualVAE import mutual_train as vae_mutual_train
from RandomAE import random_train as ae_random_train
from RandomVAE import random_train as vae_random_train
from SingleTeacherAE import single_teacher_train as single_teacher_ae_train
from SingleTeacherVAE import single_teacher_train as single_teacher_vae_train
from DoubleTeacherAE import double_teacher_train as double_teacher_ae_train
from DoubleTeacherVAE import double_teacher_train as double_teacher_vae_train

def trainAll():
    print("Training Simple AE...")
    ae_train(
        epochs=200,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

    print("Training Simple VAE...")
    vae_train(
        epochs=200,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

    print("Training Mutual Connection AE...")
    ae_mutual_train(
        pretrain_epochs=100,
        loops=50,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

    print("Training Mutual Connection VAE...")
    vae_mutual_train(
        pretrain_epochs=100,
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

    print("Training Single Teacher AE...")
    single_teacher_ae_train(
        epochs=100,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

    print("Training Single Teacher VAE...")
    single_teacher_vae_train(
        epochs=100,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

    print("Training Double Teacher AE...")
    double_teacher_ae_train(
        epochs=200,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

    print("Training Double Teacher VAE...")
    double_teacher_vae_train(
        epochs=200,
        save=True,
        scheduler_type='StepLR',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )

if __name__ == "__main__":
    trainAll()