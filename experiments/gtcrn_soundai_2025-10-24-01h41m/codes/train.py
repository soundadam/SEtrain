import os
import torch
import random
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from distributed_utils import reduce_value
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
from models.gtcrn_end2end import GTCRN as Model
from loss_factory import HybridLoss as Loss
from dataloader import DNS3Dataset as Dataset
from scheduler import LinearWarmupCosineAnnealingLR as WarmupLR
import signal
import socket
import sys

seed = 43
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True


def run(rank, config, args):
    if args.world_size > 1:
        try:
            dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
            torch.cuda.set_device(rank)
            dist.barrier()
        except Exception as e:
            print(f"Error initializing process group on rank {rank}: {e}", file=sys.stderr)
            # proceed to allow cleanup in finally
            raise

    # register quick signal cleanup inside each process
    def _sig_cleanup(signum, frame):
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_cleanup)
    signal.signal(signal.SIGTERM, _sig_cleanup)

    args.rank = rank
    args.device = torch.device(rank)
    
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    # config['train_dataloader']['batch_size'] = config['train_dataloader']['batch_size'] // args.world_size
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'],
                                                        shuffle=False,
                                                        collate_fn=collate_fn)

    # Merge NAS config into network config
    network_config = dict(config['network_config'])
    if 'nas_config' in config:
        network_config.update(config['nas_config'])

    model = Model(**network_config).to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(params=model.parameters(), **config['optimizer'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['scheduler']['kwargs'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler']['kwargs'])
    scheduler = WarmupLR(optimizer, **config['scheduler']['kwargs'])
    
    loss_func = Loss(**config['loss']).to(args.device)

    trainer = Trainer(config=config, model=model,optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
 
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)
        
        # save the config and codes
        if self.rank == 0:
            data = OmegaConf.create(config)
            OmegaConf.save(data, os.path.join(self.exp_path, 'config.yaml'))

            shutil.copy2(__file__, self.exp_path)
            for file in Path(__file__).parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            shutil.copytree(Path(__file__).parent / 'models', Path(self.code_path) / 'models', dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        # Early stopping configuration
        self.early_stopping_patience = self.trainer_config.get('early_stopping_patience', 20)
        self.early_stopping_enabled = self.trainer_config.get('early_stopping_enabled', True)
        self.epochs_without_improvement = 0
        self.early_stopped = False

        if self.resume:
            self._resume_checkpoint()

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict,
                      'best_score': self.best_score,
                      'epochs_without_improvement': self.epochs_without_improvement}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score
            self.epochs_without_improvement = 0
            if self.rank == 0:
                print(f'New best SI-SDR: {score:.4f} at epoch {epoch}')
        else:
            self.epochs_without_improvement += 1
            if self.rank == 0:
                print(f'No improvement for {self.epochs_without_improvement} epochs (best: {self.best_score:.4f})')

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

        # Restore early stopping state
        self.best_score = checkpoint.get('best_score', 0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

    def _train_epoch(self, epoch):
        total_loss = 0
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=110)

        for step, (noisy, clean) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)  
            
            enhanced = self.model(noisy)
                
            loss = self.loss_func(enhanced, clean)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            self.train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'train_loss={:.3f}'.format(total_loss / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

            if self.config['scheduler']['update_interval'] == 'step':
                self.scheduler.step()

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0
        total_sisdr_score = 0
        sisdr_metric = SISDR().to(self.device)
        self.validation_bar = tqdm(self.validation_dataloader, ncols=123)
        for step, (noisy, clean) in enumerate(self.validation_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)  
            
            enhanced = self.model(noisy)

            loss = self.loss_func(enhanced, clean)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            # Compute SI-SDR
            sisdr_score = sisdr_metric(enhanced, clean) - sisdr_metric(noisy, clean) 
            if self.world_size > 1:
                sisdr_score = reduce_value(sisdr_score)
            total_sisdr_score += sisdr_score.item()

            # Compute PESQ
            clean_np = clean.cpu().numpy()
            enhanced_np = enhanced.detach().cpu().numpy()
            # pesq_score_batch = Parallel(n_jobs=-1)(
            #     delayed(pesq)(16000, c, e, 'wb') for c, e in zip(clean, enhanced))
            # pesq_score = torch.tensor(pesq_score_batch, device=self.device).mean()
            # if self.world_size > 1:
            #     pesq_score = reduce_value(pesq_score)
            # total_pesq_score += pesq_score
            
            if self.rank == 0 and (epoch==1 or epoch %10 == 0) and step <= 3:
                noisy_path = os.path.join(self.sample_path, 'sample_{}_noisy.wav'.format(step))
                clean_path = os.path.join(self.sample_path, 'sample_{}_clean.wav'.format(step))
                enhanced_path = os.path.join(self.sample_path, 'sample_{}_enh_epoch{}.wav'.format(step, str(epoch).zfill(3)))
                if not os.path.exists(noisy_path):
                    noisy_np = noisy.cpu().numpy()
                    sf.write(noisy_path, noisy_np[0], samplerate=self.config['samplerate'])
                    sf.write(clean_path, clean_np[0], samplerate=self.config['samplerate'])

                sf.write(enhanced_path, enhanced_np[0], samplerate=self.config['samplerate'])

            self.validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'loss={:.3f},  SISDRi={:.2f}'.format(
                total_loss / step,  total_sisdr_score / step)
            # self.validation_bar.postfix = 'valid_loss={:.3f}, pesq={:.4f}'.format(
            #     total_loss / step, total_pesq_score / step)
        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        avg_loss = total_loss / step
        # avg_pesq = total_pesq_score / step
        avg_sisdr = total_sisdr_score / step

        if self.rank == 0:
            self.writer.add_scalars(
                'val_metrics', {
                    'val_loss': avg_loss,
                    # 'pesq': avg_pesq,
                    'sisdr': avg_sisdr
                }, epoch)

        return avg_loss,  avg_sisdr


    def train(self):
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
        # if epoch % validation_interval == 0:
        #     self._set_eval_mode()
        #     valid_loss, score = self._validation_epoch(epoch)
        # else:
        #     valid_loss, score = None, None

            valid_loss, score = self._validation_epoch(epoch)

            if self.config['scheduler']['update_interval'] == 'epoch':
                if self.config['scheduler']['use_plateau']:
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

            # Early stopping check
            if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                if self.rank == 0:
                    print(f'\n{"="*80}')
                    print(f'Early stopping triggered at epoch {epoch}')
                    print(f'No improvement in SI-SDR for {self.early_stopping_patience} consecutive epochs')
                    print(f'Best SI-SDR: {self.best_score:.4f} at epoch {epoch - self.epochs_without_improvement}')
                    print(f'{"="*80}\n')
                self.early_stopped = True
                break

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

            if self.early_stopped:
                print('------------Training stopped early after {} epochs------------'.format(epoch - self.start_epoch + 1))
            else:
                print('------------Training for {} epochs is done!------------'.format(self.epochs))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train.yaml')
    parser.add_argument('-N', '--nas_config', default='configs/nas_train.yaml', help='NAS configuration file')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Load NAS configuration and merge with main config
    nas_config = OmegaConf.load(args.nas_config)
    config = OmegaConf.merge(config, nas_config)
    # parser.add_argument('-D', '--device', default='0,1,2,3', help='The index of the available devices, e.g. 0,1,2,3')

    os.environ["CUDA_VISIBLE_DEVICES"] = config['DDP']['device']
    args.world_size = len(config['DDP']['device'].split(','))
    
    # If using distributed training, pick a free port to avoid EADDRINUSE from previous runs
    if args.world_size > 1:
        # find a free port by binding to port 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            free_port = s.getsockname()[1]
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(free_port)
        print(f"Using MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")
     
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
