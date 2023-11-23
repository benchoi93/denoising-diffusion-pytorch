from sklearn.preprocessing import StandardScaler
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch.utils.data import TensorDataset
import numpy as np
import wandb
wandb.init(project="ddpm_ts")

df = np.load("./PEMSBAY_2022.npy")
snesor_i = 1
data = df[:, snesor_i, 1]

# define zscore scaler
scaler = StandardScaler()
scaler.fit(np.expand_dims(data, axis=1))
data = scaler.transform(np.expand_dims(data, axis=1))
data = torch.from_numpy(data).float()

seq_len = 12*12
total, _ = data.shape
# transform to (batch-seq_len, seq_len, 1)
dataset = torch.stack([data[i:(total - seq_len + i)]
                       for i in range(seq_len)], dim=2)

model = Unet1D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_len,
    timesteps=1000,
    objective='pred_v'
)

# training_seq = torch.rand(64, 32, 128)  # features are normalized from 0 to 1
# # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below
# dataset = Dataset1D(training_seq)

# loss = diffusion(training_seq)
# loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=700000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size=4)
sampled_seq.shape  # (4, 32, 128)
