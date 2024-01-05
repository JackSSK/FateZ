import torch
import fatez as fz
from fatez.test.faker import Faker

n_gpus = torch.cuda.device_count()
print('Using GPUs:', n_gpus)
fake = Faker(device = 0)
trainer = fake.test_trainer()
tuner = fake.test_tuner(trainer = trainer)