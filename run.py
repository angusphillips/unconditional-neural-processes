import os
import socket
import logging
import time
from hydra.utils import instantiate
from logging.loggers_pl import LoggerCollection
import torch
import jax
from torch.utils.data import DataLoader, random_split
import torch.cuda as cuda
import random
import numpy as np
from tqdm import trange
from data.tensordataset import TensorDataset
from sgm.utils import ModelWrapper
from unconditional_neural_processes.neural_process import NeuralProcess
from unconditional_neural_processes.training import NeuralProcessTrainer
from unconditional_neural_processes.datasets import QuadraticData
from evaluate.utils import MetricsCollection
from evaluate.plotting import basic_plots

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
    model_file = os.path.join(ckpt_path, 'model.pt')
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
    rng = jax.random.PRNGKey(cfg.seed)
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset)

    if isinstance(dataset, TensorDataset):
        # split and wrap dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, seed=0
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    log.info("Stage : Instantiate model")
    # Create NP model
    neuralprocess = NeuralProcess(cfg.x_dim, cfg.y_dim, cfg.r_dim, cfg.z_dim, cfg.h_dim, device).to(device)

    log.info("Stage : Instantiate trainer")
    # Dataloader
    data_loader = train_ds

    # Optimiser
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=cfg.lr)

    # Trainer
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                    num_context_range=None,
                                    num_extra_target_range=None,
                                    logger=logger)

    if cfg.resume or cfg.mode == "test":  # if resume or evaluate
        neuralprocess.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    if cfg.mode == "train" or cfg.mode == "all":
        log.info("Stage : Training")
        neuralprocess.training = True
        np_trainer.train(data_loader = data_loader, epochs=cfg.epochs)
        torch.save(neuralprocess, model_file)

    model_wrapper = ModelWrapper(model=neuralprocess, cfg=cfg, device=device)
    
    if cfg.mode == "test" or (cfg.mode == "all" and success):
        log.info("Stage : Test")
        if cfg.test_val:
            metrics.get_and_log_metrics(model_wrapper, eval_ds, cfg, log, logger, "val", cfg.epochs)
        if cfg.test_test:
            metrics.get_and_log_metrics(model_wrapper, test_ds, cfg, log, logger, "test", cfg.epochs)
        if cfg.test_plot:
            basic_plots(model_wrapper, test_ds, cfg, logger, 'test', cfg.epochs)
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")