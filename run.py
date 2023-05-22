import os
from pickletools import read_long1
import socket
import logging
import time

from hydra.utils import instantiate


from score_sde.utils.loggers_pl import LoggerCollection

import torch
import jax
from torch.utils.data import DataLoader, random_split
import torch.cuda as cuda
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from neural_processes_dupont.neural_process import NeuralProcess
from neural_processes_dupont.training import NeuralProcessTrainer
from neural_processes_dupont.datasets import QuadraticData
from score_sde.utils.metrics import MetricsCollection
from score_sde.utils.plotting import oned_samples

log = logging.getLogger(__name__)

def run(cfg):
    log.info("Stage : Startup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"CUDA Devices: {cuda.current_device()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    # logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    log.info("Stage : Set seeds")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    log.info("Stage : Instantiate metrics")
    metrics = [instantiate(metric) for metric in cfg.metric.values()]
    metrics = MetricsCollection(metrics)

    log.info("Stage : Instantiate dataset")
    dataset = instantiate(cfg.dataset)

    if isinstance(dataset, QuadraticData):
        train_ds, test_ds = dataset, dataset
    else:
        lengths = [round(p * dataset.num_samples) for p in cfg.splits]
        train_ds, test_ds = random_split(dataset, lengths)

    log.info("Stage : Instantiate model")
    # Create NP model
    neuralprocess = NeuralProcess(cfg.x_dim, cfg.y_dim, cfg.r_dim, cfg.z_dim, cfg.h_dim, device).to(device)

    log.info("Stage : Instantiate trainer")
    # Dataloader
    data_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # Optimiser
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=cfg.lr)

    # Trainer
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                    num_context_range=None,
                                    num_extra_target_range=None,
                                    logger=logger)

    log.info("Stage : Training")
    neuralprocess.training = True
    t0 = time.time()
    np_trainer.train(data_loader = data_loader, epochs=cfg.epochs)
    t1 = time.time()
    log.info('training time: ', t1-t0)

    log.info("Stage : Generate samples")
    y0 = np.zeros((cfg.required_samples, dataset.num_xvals))
    y = np.zeros((cfg.required_samples, dataset.num_xvals))

    x = torch.from_numpy(np.expand_dims(np.expand_dims(dataset.xvals, 0), -1)).to(device)
    for i in trange(cfg.required_samples):
        idx = np.random.choice(len(test_ds))
        _, y0_ = test_ds[idx]
        z = torch.randn((1, cfg.z_dim)).to(device)
        y_, _ = neuralprocess.xz_to_y(x.float(), z.float())
        y0[i,:] = y0_.flatten().cpu().detach().numpy()
        y[i,:] = y_.flatten().cpu().detach().numpy()

    # Save samples
    np.save(os.path.join(ckpt_path, 'x.npy'), x.cpu().detach().numpy())
    np.save(os.path.join(ckpt_path, 'y.npy'), y)
    np.save(os.path.join(ckpt_path, 'y0.npy'), y0)

    # Plot
    # log.info("Stage : Plotting")
    # fig = oned_samples(None, dataset.xvals, y0, y, dataset, cfg.plotting.num_samples_vis, model_samples=True, dataset_samples=True)
    # logger.log_plot('model_dataset_samples', fig, cfg.epochs)
    # plt.close(fig)

    # fig = oned_samples(None, dataset.xvals, y0, y, dataset, cfg.plotting.num_samples_vis, model_samples=False, dataset_samples=True)
    # logger.log_plot('dataset_samples', fig, cfg.epochs)
    # plt.close(fig)

    # fig = oned_samples(None, dataset.xvals, y0, y, dataset, cfg.plotting.num_samples_vis, model_samples=True, dataset_samples=False)
    # logger.log_plot('model_samples', fig, cfg.epochs)
    # plt.close(fig)

    log.info("Stage : Computing metric")
    stage = 'test'
    rng = jax.random.PRNGKey(cfg.seed)
    for metric in metrics.metrics_list:
        power, mmd2 = 0.0, 0.0
        n_it = int(metric.n_tests/metric.batch)
        nb = metric.batch*metric.n_samples
        for i in trange(n_it):
            rng, rng1 = jax.random.split(rng, num=2)

            power_, mmd2_ = metric.power_test(y0[(i*nb):((i+1)*nb),:], y[(i*nb):((i+1)*nb),:], rng1)
            power += power_/n_it
            mmd2 += mmd2_/n_it

        logger.log_metrics({f"{stage}/power_{metric.kernel_T}": power}, cfg.epochs)
        log.info(f"{stage}/power_{metric.kernel_T} = {power:.3f}")

        logger.log_metrics({f"{stage}/mmd2_{metric.kernel_T}": mmd2}, cfg.epochs)
        log.info(f"{stage}/mmd2_{metric.kernel_T} = {mmd2:.5f}")