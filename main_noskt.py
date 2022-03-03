import json
import math
import pdb
from decimal import Decimal

import cv2
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from ssim import SSIM
# import data
import data_new
import model
import utility
# from model.edsr import PAMS_EDSR
# from model.dorefa_edsr import PAMS_EDSR
# from model.edsr_org import EDSR
# from model.rdn import PAMS_RDN
# from model.rdn_org import RDN
from option import args
from utils import common as util
from utils.common import AverageMeter, load_check
from importlib import import_module
# from model.tflite_edsr import PAMS_EDSR
if args.model.lower() == 'edsr':
    PAMS_EDSR = getattr(import_module(f'model.{args.model_name}_edsr'),'PAMS_EDSR')
elif args.model.lower() == 'rdn':
    PAMS_RDN = getattr(import_module(f'model.{args.model_name}_rdn'),'PAMS_RDN')
elif args.model.lower() == 'srresnet':
    PAMS_SRResNet = getattr(import_module(f'model.{args.model_name}_srresnet'),'SRResNet')

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda')

class Trainer():
    def __init__(self, args, loader, s_model, ckp, quantized_ckp=None):
        self.args = args
        self.scale = args.scale
        self.ssim = SSIM()
        self.epoch = 0
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        # self.t_model = t_model
        self.s_model = s_model
        arch_param = [v for k, v in self.s_model.named_parameters() if 'alpha' not in k]
        alpha_param = [v for k, v in self.s_model.named_parameters() if 'alpha' in k]

        if quantized_ckp is None:
            params = [{'params': arch_param}, {'params': alpha_param, 'lr': 1e-2}]
        else:
            self.epoch = 20
            arch_param = []
            arch_trained_param = []
            for k, v in self.s_model.named_parameters():
                if 'alpha' not in k:
                    if k not in quantized_ckp:
                        arch_param.append(v)
                    else:
                        arch_trained_param.append(v)
            alpha_param = [v for k, v in self.s_model.named_parameters() if 'alpha' in k]

            print('arch_trained_param (should be zero lr)', len(arch_trained_param), 'arch_param', len(arch_param))
            params = [{'params': arch_param}, {'params': alpha_param, 'lr': 0}, {'params': arch_trained_param, 'lr': 0}]

        self.optimizer = torch.optim.Adam(params, lr=args.lr, betas=args.betas, eps=args.epsilon)
        self.sheduler = StepLR(self.optimizer, step_size=int(args.decay), gamma=args.gamma)
        self.writer_train = SummaryWriter(ckp.dir + '/run/train')

        if args.resume is not None:
            ckpt = torch.load(args.resume)
            self.epoch = ckpt['epoch']
            print(f"Continue from {self.epoch}")
            self.s_model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.sheduler.load_state_dict(ckpt['scheduler'])

        self.losses = AverageMeter()
        self.att_losses = AverageMeter()
        self.nor_losses = AverageMeter()


    def train(self):

        self.epoch = self.epoch + 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        self.writer_train.add_scalar(f'lr', lr, self.epoch)
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch, Decimal(lr))
        )

        # self.t_model.eval()
        self.s_model.train()

        self.s_model.apply(lambda m: setattr(m, 'epoch', self.epoch))

        num_iterations = len(self.loader_train)
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, _,) in enumerate(self.loader_train):

            # if self.epoch ==1 and batch == 10:
            #     break


            # self.s_model.apply(lambda m: setattr(m, 'error', 0))

            num_iters = num_iterations * (self.epoch - 1) + batch

            lr, hr = self.prepare(lr, hr)
            data_size = lr.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            if hasattr(self.s_model, 'set_scale'):
                self.s_model.set_scale(idx_scale)

            s_sr, s_res = self.s_model(lr)


            nor_loss = args.w_l1 * F.l1_loss(s_sr, hr)

            loss = nor_loss

            if torch.any(torch.isnan(loss)):
                print('None loss!!')
                pdb.set_trace()


            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            self.losses.update(loss.item(), data_size)
            display_loss = f'Loss: {self.losses.avg: .3f}'

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    display_loss,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            for name, value in self.s_model.named_parameters():
                if 'alpha' in name:
                    # if value.grad is not None:
                    if value.grad is not None and value.grad.squeeze().ndim == 0:
                        self.writer_train.add_scalar(f'{name}_grad', value.grad.cpu().data.numpy(), num_iters)
                        self.writer_train.add_scalar(f'{name}_data', value.cpu().data.numpy(), num_iters)

        self.sheduler.step()

    def test(self, is_teacher=False):
        torch.set_grad_enabled(False)

        self.s_model.apply(lambda m: setattr(m, 'test_only', args.test_only))
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        if is_teacher:
            model = self.t_model
        else:
            model = self.s_model
        model.eval()
        timer_test = utility.timer()

        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            
            for idx_scale, scale in enumerate(self.scale):
                ssim_mean = 0
                d.dataset.set_scale(idx_scale)
                i = 0
                # for lr, hr, filename, _ in tqdm(d, ncols=80):
                for lr, hr, filename in tqdm(d, ncols=80):
                    i += 1
                    lr, hr = self.prepare(lr, hr)
                    sr, s_res = model(lr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    cur_psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    cur_ssim = self.ssim(sr,hr)
                    ssim_mean += cur_ssim
                    self.ckp.log[-1, idx_data, idx_scale] += cur_psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        save_name = f'{args.k_bits}bit_{filename[0]}'
                        self.ckp.save_results(d, save_name, save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                ssim_mean /= len(d)
                self.ckp.write_log(
                    '[{} x{}] PSNR: {:.3f}  (Best: {:.3f} @epoch {})  SSIM: {:.3f}'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        ssim_mean
                    )
                )
                self.writer_train.add_scalar(f'psnr', self.ckp.log[-1, idx_data, idx_scale], self.epoch)



        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            is_best = (best[1][0, 0] + 1 == epoch)
            state = {
                'epoch': epoch,
                'state_dict': self.s_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.sheduler.state_dict()
            }
            util.save_checkpoint(state, is_best, checkpoint=self.ckp.dir + '/model')
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.cuda()

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs


def main():
    if checkpoint.ok:
        loader = data_new.Data(args)
        if args.model.lower() == 'edsr':
            # t_model = EDSR(args, is_teacher=True).to(device)
            s_model = PAMS_EDSR(args, bias=True).to(device)
        elif args.model.lower() == 'rdn':
            # t_model = RDN(args, is_teacher=True).to(device)
            s_model = PAMS_RDN(args).to(device)
        elif args.model.lower() == 'srresnet':
            # t_model = RDN(args, is_teacher=True).to(device)
            s_model = PAMS_SRResNet(args).to(device)
        else:
            raise ValueError('not expected model = {}'.format(args.model))

        if args.pre_train is not None:

            s_model_dict = s_model.state_dict()
            s_model.load_state_dict(s_model_dict)


        # quantized model load the pre-trained quantized parameter

        quanztied_ckpt = None

        if args.test_only:
            if args.refine is None:
                ckpt = torch.load(f'{args.save}/model/model_best.pth.tar')
                refine_path = f'{args.save}/model/model_best.pth.tar'
            else:
                ckpt = torch.load(f'{args.refine}')
                refine_path = args.refine

            s_checkpoint = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            for k, v in s_checkpoint.items():
                if 'max_val' in k or 'min_val' in k:
                    s_checkpoint[k] = torch.reshape(v, torch.ones(1).shape)
                    # s_checkpoint = {k:v for k,v in s_checkpoint.items() if 'max_val' not in k}

            s_model.load_state_dict(s_checkpoint)
            print(f"Load model from {refine_path}")

        t = Trainer(args, loader, s_model, checkpoint, quanztied_ckpt)
        if quanztied_ckpt is not None:
            t.test()
        print(f'{args.save} start!')
        while not t.terminate():
            # t.test(True)
            t.train()
            t.test()

        checkpoint.done()
        print(f'{args.save} done!')


if __name__ == '__main__':
    main()
