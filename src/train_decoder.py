# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data import DatasetSet
from wavenet import WaveNet
from wavenet_models import cross_entropy_loss, Encoder
from utils import create_output_dir, LossMeter, wrap

parser = argparse.ArgumentParser(description='PyTorch Code for A Universal Music Translation Network')
# Env options:
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, required=True,
                    help='Experiment name')
parser.add_argument('--data',
                    metavar='D', type=Path, help='Data path', nargs='+')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--load-optimizer', action='store_true')
parser.add_argument('--per-epoch', action='store_true',
                    help='Save model per epoch')

# Data options
parser.add_argument('--seq-len', type=int, default=16000,
                    help='Sequence length')
parser.add_argument('--epoch-len', type=int, default=10000,
                    help='Samples per epoch')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--num-workers', type=int, default=10,
                    help='DataLoader workers')
parser.add_argument('--data-aug', action='store_true',
                    help='Turns data aug on')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Data augmentation magnitude.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=0.98,
                    help='new LR = old LR * decay')
parser.add_argument('--short', action='store_true',
                    help='Run only a few batches per epoch for testing')
parser.add_argument('--h5-dataset-name', type=str, default='wav',
                    help='Dataset name in .h5 file')

# Encoder options
parser.add_argument('--latent-d', type=int, default=128,
                    help='Latent size')
parser.add_argument('--repeat-num', type=int, default=6,
                    help='No. of hidden layers')
parser.add_argument('--encoder-channels', type=int, default=128,
                    help='Hidden layer size')
parser.add_argument('--encoder-blocks', type=int, default=3,
                    help='No. of encoder blocks.')
parser.add_argument('--encoder-pool', type=int, default=800,
                    help='Number of encoder outputs to pool over.')
parser.add_argument('--encoder-final-kernel-size', type=int, default=1,
                    help='final conv kernel size')
parser.add_argument('--encoder-layers', type=int, default=10,
                    help='No. of layers in each encoder block.')
parser.add_argument('--encoder-func', type=str, default='relu',
                    help='Encoder activation func.')

# Decoder options
parser.add_argument('--blocks', type=int, default=4,
                    help='No. of wavenet blocks.')
parser.add_argument('--layers', type=int, default=10,
                    help='No. of layers in each block.')
parser.add_argument('--kernel-size', type=int, default=2,
                    help='Size of kernel.')
parser.add_argument('--residual-channels', type=int, default=128,
                    help='Residual channels to use.')
parser.add_argument('--skip-channels', type=int, default=128,
                    help='Skip channels to use.')
parser.add_argument('--grad-clip', type=float,
                    help='If specified, clip gradients with specified magnitude')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.n_datasets = len(self.args.data)
        self.expPath = Path('checkpoints') / args.expName

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.logger = create_output_dir(args, self.expPath)
        self.data = [DatasetSet(d, args.seq_len, args) for d in args.data]

        self.losses_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.loss_total = LossMeter('total')

        self.evals_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.eval_total = LossMeter('eval total')

        self.encoder = Encoder(args)
        self.decoder = WaveNet(args)

        assert args.checkpoint, 'you MUST pass a checkpoint for the encoder'
        self.start_epoch = 0
        states = torch.load(args.checkpoint)
        self.encoder.load_state_dict(states['encoder_state'])
        self.decoder.load_state_dict(states['decoder_state'])
        self.logger.info('Loaded checkpoint parameters')

        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.decoder = torch.nn.DataParallel(self.decoder).cuda()

        self.model_optimizer = optim.Adam(self.decoder.parameters(), lr=args.lr)

        self.lr_manager = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, args.lr_decay)
        self.lr_manager.last_epoch = self.start_epoch
        self.lr_manager.step()

    def eval_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        z = self.encoder(x)
        y = self.decoder(x, z)

        recon_loss = cross_entropy_loss(y, x)
        self.evals_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        total_loss = recon_loss.mean().data.item()
        self.eval_total.add(total_loss)

        return total_loss

    def train_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        # optimize G - reconstructs well
        z = self.encoder(x_aug)
        z = z.detach()  # stop gradients
        y = self.decoder(x, z)

        recon_loss = cross_entropy_loss(y, x)
        self.losses_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        loss = recon_loss.mean()
        self.model_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.decoder.parameters(), self.args.grad_clip)
        self.model_optimizer.step()

        self.loss_total.add(loss.data.item())

        return loss.data.item()

    def train_epoch(self, epoch):
        for meter in self.losses_recon:
            meter.reset()
        self.loss_total.reset()

        self.encoder.eval()
        self.decoder.train()

        n_batches = self.args.epoch_len

        with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 3:
                    break

                dset_num = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dset_num].train_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                batch_loss = self.train_batch(x, x_aug, dset_num)

                train_enum.set_description(f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def evaluate_epoch(self, epoch):
        for meter in self.evals_recon:
            meter.reset()
        self.eval_total.reset()

        self.encoder.eval()
        self.decoder.eval()

        n_batches = int(np.ceil(self.args.epoch_len / 10))

        with tqdm(total=n_batches) as valid_enum, \
                torch.no_grad():
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 10:
                    break

                dset_num = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dset_num].valid_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                batch_loss = self.eval_batch(x, x_aug, dset_num)

                valid_enum.set_description(f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
                valid_enum.update()

    @staticmethod
    def format_losses(meters):
        losses = [meter.summarize_epoch() for meter in meters]
        return ', '.join('{:.4f}'.format(x) for x in losses)

    def train_losses(self):
        meters = [*self.losses_recon]
        return self.format_losses(meters)

    def eval_losses(self):
        meters = [*self.evals_recon]
        return self.format_losses(meters)

    def train(self):
        best_eval = float('inf')

        # Begin!
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f'Starting epoch, Rank {self.args.rank}, Dataset: {self.args.data[self.args.rank]}')
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            self.logger.info(f'Epoch %s Rank {self.args.rank} - Train loss: (%s), Test loss (%s)',
                             epoch, self.train_losses(), self.eval_losses())
            self.lr_manager.step()
            val_loss = self.eval_total.summarize_epoch()

            if val_loss < best_eval:
                self.save_model(f'bestmodel_{self.args.rank}.pth')
                best_eval = val_loss

            if not self.args.per_epoch:
                self.save_model(f'lastmodel_{self.args.rank}.pth')
            else:
                self.save_model(f'lastmodel_{epoch}_rank_{self.args.rank}.pth')

            torch.save([self.args,
                        epoch],
                        '%s/args.pth' % self.expPath)

            self.logger.debug('Ended epoch')

    def save_model(self, filename):
        save_path = self.expPath / filename

        states = torch.load(self.args.checkpoint)

        torch.save({'encoder_state': states['encoder_state'],
                    'decoder_state': self.decoder.module.state_dict(),
                    'model_optimizer_state': self.model_optimizer.state_dict(),
                    'dataset': self.args.rank,
                    },
                   save_path)

        self.logger.debug(f'Saved model to {save_path}')


def main():
    args = parser.parse_args()
    args.rank = 0

    Trainer(args).train()


if __name__ == '__main__':
    main()
