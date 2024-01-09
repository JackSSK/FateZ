import torch
import fatez as fz
import fatez.model as model
import torch.multiprocessing as mp
from fatez.test.faker import Faker


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    train_epoch = 20
    tune_epoch = 10
    trainer_save_path = 'results/faker.ckpt'
    tuner_save_path = 'results/faker.ckpt'
    print('Using GPUs:', n_gpus)

    fake = Faker(
        world_size = n_gpus,
        device = list(range(n_gpus)),
        # quiet = False,
    )

    print('Testing Trainer Model.\n')
    mp.spawn(
        fake.test_trainer_main,
        args = (
            n_gpus,
            train_epoch,
            trainer_save_path,
        ),
        #  total number of processes - # gpus
        nprocs = n_gpus, 
    )
    print('Pre-Trainer OK.\n')

    print('Testing Tuner Model.\n')
    mp.spawn(
        fake.test_tuner_main,
        args = (
            n_gpus,
            trainer_save_path,
            train_epoch,
            tuner_save_path,
        ),
        #  total number of processes - # gpus
        nprocs = n_gpus, 
    )
    print('Fine-Tuner OK.\n')

    print('Testing Explainer.\n')
    fake.test_explainer(
        rank = 0,
        world_size = 1,
        tuner_path = tuner_save_path,
    )
    print('Explainer OK.\n')


