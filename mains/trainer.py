from models.networks import ResAdaInGen, GANLoss, Unet_Discriminator, smri2scalarGen, q_DL, q_DL2D
from utils.torchutils import weights_init, get_scheduler
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np

####################################
# Proposed: sMRI to DWIs
###################################
class dwi_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(dwi_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.multimodal = hyperparameters['multimodal_t1'] or hyperparameters['multimodal_t2']
        self.mb0 = hyperparameters['multimodal_b0'] > 0
        self.mt1 = hyperparameters['multimodal_t1'] > 0
        self.mt2 = hyperparameters['multimodal_t2'] > 0
        assert hyperparameters['multimodal_b0'] + hyperparameters['multimodal_t1'] + hyperparameters['multimodal_t2'] == hyperparameters['input_dim']
        print(hyperparameters['gen']['g_type'])
        if hyperparameters['gen']['g_type'] == 'resnet':
            self.gen_a = ResAdaInGen(hyperparameters['input_dim'], hyperparameters['output_dim'],hyperparameters['gen'])  # auto-encoder for domain a
        else:
            raise NotImplementedError

        self.style_dim = hyperparameters['gen']['style_dim']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        all = sum([np.prod(p.size()) for p in self.gen_a.parameters()])
        gen_params = list(self.gen_a.parameters())
        trainable = sum([np.prod(p.size()) for p in gen_params if p.requires_grad])
        print('G Trainable: {}/{}M'.format(trainable/1000000, all/1000000))

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        gpu_ids = hyperparameters['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu') # get device name: CPU or GPU
        print('Deploy to {}'.format(self.device))
        self.gen_a.to(self.device)
        # Network weight initialization
        self.gen_a.apply(weights_init(hyperparameters['init']))
        print('Init generator with {}'.format(hyperparameters['init']))
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.loss_dwi = torch.zeros([]).to(self.device) # L1 loss initialize
        self.l1_w = hyperparameters['l1_w'] #L1 weight
        self.gan_w = hyperparameters['gan_w'] #GAN weight


        if self.gan_w > 0:
            print('GAN with {} discriminator'.format(hyperparameters['dis']['d_type']))
            self.dis_type = hyperparameters['dis']['d_type']
            self.dis = Unet_Discriminator(hyperparameters['dis']['in_dim'], ndf=hyperparameters['dis']['dim'],
                                          n_layers=hyperparameters['dis']['n_layer'], n_latent=2, embed=True, device=self.device)

            self.dis_opt = torch.optim.Adam([p for p in list(self.dis.parameters()) if p.requires_grad],
                                            lr=hyperparameters['dis']['lr_d'],
                                            betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
            self.criterionGAN = GANLoss(hyperparameters['dis']['gan_type'])
            self.dis.to(self.device)
            self.dis.apply(weights_init(hyperparameters['init']))
            trainable = sum([np.prod(p.size()) for p in self.dis.parameters() if p.requires_grad])
            all = sum([np.prod(p.size()) for p in self.dis.parameters()])
            print('D Trainable: {}/{}M'.format(trainable/1000000, all/1000000))
        else:
            self.dis_scheduler = None

    def recon_criterion(self, input, target, exp=False):
        brain_mask = 1.-(target==0.).float()
        if exp: pix_loss =  F.l1_loss(torch.exp(input), torch.exp(target), reduction='sum')
        else: pix_loss =  F.l1_loss(input, target, reduction='sum')
        pix_loss = pix_loss / (brain_mask.sum() + 1e-10)
        return pix_loss

    def forward(self, b0, bvec):
        self.eval()
        x_img = self.gen_a.forward(b0, bvec)
        self.train()
        return x_img

    def sample(self, data_dict):
        self.gen_a.eval()
        return_dict, in_i = self.prepare_input(data_dict, 0)
        cond_i = data_dict['cond_1'].to(self.device).float()
        dwi_i = data_dict['dwi_1'].to(self.device).float()
        pred_i = self.gen_a.forward(in_i, cond_i)
        loss_dwi = self.recon_criterion(pred_i, dwi_i, False)
        self.gen_a.train()
        return_dict['dwi'] = dwi_i[0, 0].cpu().numpy()
        bvec_vis = cond_i[0].cpu().numpy()
        return_dict['pred%.1f,%.1f,%.1f[%.1f]' % (bvec_vis[0], bvec_vis[1], bvec_vis[2], bvec_vis[3])] = pred_i[0,0].detach().cpu().numpy()
        return_dict['loss'] = loss_dwi.item()
        return return_dict

    def prepare_input(self, data_dict, i):
        nc = 0
        in_i = None
        return_dict = {}
        return_dict['b0'] = data_dict['b0_%d' % (i + 1)][0,0].detach().cpu().numpy()
        if self.mb0:
            nc += 1
            in_i = data_dict['b0_%d' % (i + 1)].to(self.device).float()
        if self.mt2:
            nc += 1
            t2_i = data_dict['t2_%d' % (i + 1)].to(self.device).float()
            return_dict['t2'] = t2_i[0, 0].detach().cpu().numpy()
            if in_i is None:
                in_i = t2_i
            else:
                in_i = torch.cat((in_i, t2_i), dim=1)
        if self.mt1:
            nc += 1
            t1_i = data_dict['t1_%d' % (i + 1)].to(self.device).float()
            return_dict['t1'] = t1_i[0, 0].detach().cpu().numpy()
            if in_i is None:
                in_i = t1_i
            else:
                in_i = torch.cat((in_i, t1_i), dim=1)
        assert nc >= 1, 'wrong input setting'
        return return_dict, in_i

    def update(self, data_dict, n_dwi,  iterations):
        self.gen_opt.zero_grad()
        self.loss_dwi = torch.zeros([]).to(self.device)
        self.loss_g = torch.zeros([]).to(self.device)
        self.loss_d = torch.zeros([]).to(self.device)
        i = 0
        return_dict, in_i = self.prepare_input(data_dict, i)
        cond_i = data_dict['cond_%d'%(i+1)].to(self.device).float()
        dwi_i = data_dict['dwi_%d'%(i+1)].to(self.device).float()
        pred_i = self.gen_a.forward(in_i, cond_i)
        self.loss_dwi += self.l1_w * self.recon_criterion(pred_i, dwi_i, False)
        total_loss = self.loss_dwi
        if self.gan_w > 0:
            if self.dis_type == 'unet':
                self.loss_G_global, self.loss_G_local = self.dis.get_G_loss(in_i, pred_i, cond_i, self.criterionGAN)
                self.loss_g += 0.5 * (self.loss_G_global + self.loss_G_local) * self.gan_w
            else:
                self.loss_g += self.dis.get_G_loss(in_i, pred_i, cond_i, self.criterionGAN)
            self.set_requires_grad([self.dis], False)   # Ds require no gradients when optimizing Gs
            self.set_requires_grad([self.gen_a], True)  # Ds require no gradients when optimizing Gs
            total_loss += self.loss_g
        total_loss.backward()
        self.gen_opt.step()
        return_dict['dwi'] = dwi_i[0,0].cpu().numpy()
        bvec_vis = cond_i[0].cpu().numpy()
        return_dict['pred%.1f,%.1f,%.1f[%.1f]'%(bvec_vis[0], bvec_vis[1], bvec_vis[2],bvec_vis[3])]= pred_i[0,0].detach().cpu().numpy()

        if self.gan_w > 0:
            if iterations %2 ==0:
                self.set_requires_grad([self.dis], True)     # Ds require no gradients when optimizing Gs
                self.set_requires_grad([self.gen_a], False)  # Ds require no gradients when optimizing Gs
                self.dis_opt.zero_grad()
                select_j_list = list(set(np.arange(n_dwi)) - {0})  # unpaired
                j = select_j_list[np.random.randint(len(select_j_list))]
                _, in_j = self.prepare_input(data_dict, j)
                dwi_j = data_dict['dwi_%d' % (j + 1)].to(self.device).float()
                cond_j = data_dict['cond_%d' % (j + 1)].to(self.device).float()
                self.loss_D_global, self.loss_D_local = self.dis.get_D_loss(in_j, dwi_j, cond_j, in_i, pred_i, cond_i, self.criterionGAN)
                print('Global D: {}, Local D: {}'.format(self.loss_D_global.item(), self.loss_D_local.item()))
                self.loss_d = 0.5 * (self.loss_D_global + self.loss_D_local) * self.gan_w

                self.loss_d.backward()
                self.dis_opt.step()

        return return_dict


    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def resume(self, checkpoint_dir):
        # Load generators
        print('Load model G from %s'%checkpoint_dir)
        state_dict = torch.load(checkpoint_dir)
        self.gen_a.load_state_dict(state_dict['a'])
        # Load optimizers
        last_opt_name = checkpoint_dir.replace("gen", "opt")
        print('Load G optimizer: %s'%last_opt_name)
        state_dict = torch.load(last_opt_name)
        self.gen_opt.load_state_dict(state_dict['gen'])
        if self.gan_w > 0:
            print('Load D optimizer: %s' % last_opt_name)
            self.dis_opt.load_state_dict(state_dict['dis'])
            print('Load D model: %s' % checkpoint_dir.replace("gen", "dis"))
            dis_dict = torch.load(checkpoint_dir.replace("gen", "dis"))
            self.dis.load_state_dict(dis_dict['dis'])
        return


    def save(self, snapshot_dir, epoch, step=-1, gen_name=None):
        # Save generators, discriminators, and optimizers
        if gen_name is None:
            if epoch == -1:
                gen_name = os.path.join(snapshot_dir, 'gen_latest.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_latest.pt')
            else:
                if step == -1:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%d.pt' % (epoch + 1))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%d.pt' % (epoch + 1))
                else:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%dstep%d.pt' % (epoch + 1, step))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%dstep%d.pt' % (epoch + 1, step))
        else:
            opt_name = gen_name.replace('gen','opt')

        if self.gan_w > 0:
            # save optimizer for both D and G
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
            # save discriminator if using gan
            torch.save({'dis': self.dis.state_dict()}, gen_name.replace('gen','dis'))
        else:
            # save optimizer for generator only
            torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
        # save generator
        torch.save({'a': self.gen_a.state_dict()}, gen_name)


####################################
# Baseline : sMRI to scalars
###################################
class smri2scalar_Trainer(nn.Module):
    def __init__(self, hyperparameters,input_dim, output_dim):
        super(smri2scalar_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.multimodal = hyperparameters['multimodal_t1'] or hyperparameters['multimodal_t2']
        self.t1 = hyperparameters['multimodal_t1'] > 0
        self.t2 = hyperparameters['multimodal_t2'] > 0
        assert 1 + hyperparameters['multimodal_t1'] + hyperparameters['multimodal_t2'] == hyperparameters['input_dim']
        self.gen_a = smri2scalarGen(input_dim, output_dim, hyperparameters['gen'])  # auto-encoder for domain a

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen_a.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        gpu_ids = hyperparameters['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu') # get device name: CPU or GPU
        print('Deploy to {}'.format(self.device))
        self.gen_a.to(self.device)
        # Network weight initialization
        self.gen_a.apply(weights_init(hyperparameters['init']))
        print('Init generator with {}'.format(hyperparameters['init']))

        self.loss_translate = torch.zeros([]).to(self.device)
        self.l1_w = hyperparameters['l1_w']

    def recon_criterion(self, input, target, brain_mask):
        pix_loss =  F.l1_loss(input, target, reduction='sum')
        pix_loss = pix_loss / (brain_mask.sum() + 1e-10)
        loss = pix_loss
        return loss

    def forward(self, b0):
        self.eval()
        x_img = self.gen_a.forward(b0)
        self.train()
        return x_img

    def prepare_data(self, data_dict):
        return_dict = {}
        in_b0 = data_dict['b0'].to(self.device).float()
        return_dict['in_b0'] = in_b0[0, 0].cpu().numpy()
        inputs = in_b0
        if self.t2:
            in_t2 = data_dict['t2'].to(self.device).float()
            inputs = torch.cat((inputs, in_t2), dim=1)
            return_dict['in_t2'] = in_t2[0, 0].cpu().numpy()
        if self.t1:
            in_t1 = data_dict['t1'].to(self.device).float()
            inputs = torch.cat((inputs, in_t1), dim=1)
            return_dict['in_t1'] = in_t1[0, 0].cpu().numpy()
        return inputs.to(self.device).float(), return_dict

    def sample(self, data_dict, trgs):
        self.gen_a.eval()
        in_i, return_dict = self.prepare_data(data_dict)  # torch.cat((in_b0, in_t2), dim=1)
        pred_i = self.gen_a.forward(in_i)
        target = None
        for ti in range(len(trgs)):
            return_dict['trg_%s' % trgs[ti]] = data_dict[trgs[ti]][0, 0].cpu().numpy()
            if target is None:
                target = data_dict[trgs[ti]].to(self.device).float()
            else:
                target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        loss_derives = self.recon_criterion(pred_i, target, in_i.sum(dim=1)>0.)
        self.gen_a.train()
        for ti in range(len(trgs)):
            return_dict[trgs[ti]] = pred_i[0, ti].detach().cpu().numpy()
        return_dict['loss'] = loss_derives.item()
        return return_dict

    def update(self, data_dict, iterations, trgs):
        self.gen_opt.zero_grad()
        self.loss_derives = torch.zeros([]).to(self.device)
        self.loss_g = torch.zeros([]).to(self.device)
        self.loss_d = torch.zeros([]).to(self.device)
        in_i, return_dict = self.prepare_data(data_dict)#torch.cat((in_b0, in_t2),dim=1)

        target = None
        for ti in range(len(trgs)):
            return_dict['trg_%s' % trgs[ti]] = data_dict[trgs[ti]][0, 0].cpu().numpy()
            if target is None:
                target = data_dict[trgs[ti]].to(self.device).float()
            else:
                target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        pred_i = self.gen_a.forward(in_i)

        self.loss_derives += self.l1_w * self.recon_criterion(pred_i, target, in_i.sum(dim=1)>0.)
        total_loss = self.loss_derives
        total_loss.backward()
        self.gen_opt.step()

        for ti in range(len(trgs)):
            return_dict[trgs[ti]] = pred_i[0, ti].detach().cpu().numpy()
        return return_dict

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        print('Load models from %s'%checkpoint_dir)
        state_dict = torch.load(checkpoint_dir)
        self.gen_a.load_state_dict(state_dict['a'])
        # Load optimizers
        last_opt_name = checkpoint_dir.replace("gen", "opt")
        state_dict = torch.load(last_opt_name)
        self.gen_opt.load_state_dict(state_dict['gen'])
        if self.gan_w > 0:
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.dis.load_state_dict(torch.load(checkpoint_dir.replace("gen","dis")))
        return


    def save(self, snapshot_dir, epoch, step=-1, gen_name=None):
        # Save generators, discriminators, and optimizers
        if gen_name is None:
            if epoch == -1:
                gen_name = os.path.join(snapshot_dir, 'gen_latest.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_latest.pt')
            else:
                if step == -1:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%d.pt' % (epoch + 1))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%d.pt' % (epoch + 1))
                else:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%dstep%d.pt' % (epoch + 1, step))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%dstep%d.pt' % (epoch + 1, step))
        else:
            opt_name = gen_name.replace('gen','opt')

        if self.gan_w > 0:
            # save optimizer for both D and G
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
            # save discriminator if using gan
            torch.save({'dis': self.dis.state_dict()}, gen_name.replace('gen','dis'))
        else:
            # save optimizer for generator only
            torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
        # save generator
        torch.save({'a': self.gen_a.state_dict()}, gen_name)


####################################
# Baseline : qDL
###################################
class qDL_Trainer(nn.Module):
    def __init__(self, hyperparameters, input_dim, output_dim):
        super(qDL_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = q_DL(input_dim, output_dim)  # auto-encoder for domain a

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen_a.parameters())
        self.gen_opt = torch.optim.SGD([p for p in gen_params if p.requires_grad],
                                        lr=lr, momentum=0.9)

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        gpu_ids = hyperparameters['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu') # get device name: CPU or GPU
        print('Deploy to {}'.format(self.device))
        self.gen_a.to(self.device)
        # Network weight initialization
        self.gen_a.apply(weights_init('orthogonal'))
        print('Init generator with {}'.format(hyperparameters['init']))

        self.loss_derives = torch.zeros([]).to(self.device)
        self.l1_w = hyperparameters['l1_w']

    def recon_criterion(self, input, target):
        brain_mask = 1.-(target==0.).float()
        pix_loss =  F.l1_loss(input, target, reduction='sum')
        pix_loss = pix_loss / (brain_mask.sum() + 1e-10)
        loss = pix_loss
        return loss

    def forward(self, b0):
        self.eval()
        x_img = self.gen_a.forward(b0)
        self.train()
        return x_img

    def sample(self, data_dict, trgs):
        self.gen_a.eval()
        return_dict = {}
        in_i = data_dict['dwi'].to(self.device).float()
        pred_i = self.gen_a.forward(in_i)
        target = None
        for ti in range(len(trgs)):
            return_dict['trg_%s' % trgs[ti]] = data_dict[trgs[ti]][0, 0].cpu().numpy()
            if target is None:
                target = data_dict[trgs[ti]].to(self.device).float()
            else:
                target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        loss_derives = self.recon_criterion(pred_i, target)
        self.gen_a.train()
        return_dict['dwi'] = in_i[0, 0].cpu().numpy()
        for ti in range(len(trgs)):
            return_dict[trgs[ti]] = pred_i[0,ti].detach().cpu().numpy()
        return_dict['loss'] = loss_derives.item()
        return return_dict


    def update(self, data_dict, epochs, trgs):
        if epochs < 10:
            for param_group in self.gen_opt.param_groups:
                param_group['lr'] = 0.001
        self.gen_opt.zero_grad()
        in_i = data_dict['dwi'].to(self.device).float()
        target = None
        return_dict = {}
        for ti in range(len(trgs)):
            return_dict['trg_%s'%trgs[ti]] = data_dict[trgs[ti]][0,0].cpu().numpy()
            if target is None: target = data_dict[trgs[ti]].to(self.device).float()
            else: target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        pred_i = self.gen_a.forward(in_i)
        self.loss_derives = self.recon_criterion(pred_i, target)  # nn.functional.mse_loss(pred_i, target)
        self.loss_derives.backward()
        self.gen_opt.step()
        return_dict['dwi'] = in_i[0, 0].cpu().numpy()
        for ti in range(len(trgs)):
            return_dict['pred_%s'%trgs[ti]] = pred_i[0, ti].detach().cpu().numpy()
            print(trgs[ti], return_dict['pred_%s'%trgs[ti]].min(),return_dict['pred_%s'%trgs[ti]].max(), return_dict['trg_%s'%trgs[ti]].min(), return_dict['trg_%s'%trgs[ti]].max())
        return return_dict

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        print('Load models from %s'%checkpoint_dir)
        state_dict = torch.load(checkpoint_dir)
        self.gen_a.load_state_dict(state_dict['a'])
        # Load optimizers
        last_opt_name = checkpoint_dir.replace("gen", "opt")
        state_dict = torch.load(last_opt_name)
        self.gen_opt.load_state_dict(state_dict['gen'])
        return


    def save(self, snapshot_dir, epoch, step=-1, gen_name=None):
        # Save generators, discriminators, and optimizers
        if gen_name is None:
            if epoch == -1:
                gen_name = os.path.join(snapshot_dir, 'gen_latest.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_latest.pt')
            else:
                if step == -1:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%d.pt' % (epoch + 1))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%d.pt' % (epoch + 1))
                else:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%dstep%d.pt' % (epoch + 1, step))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%dstep%d.pt' % (epoch + 1, step))
        else:
            opt_name = gen_name.replace('gen','opt')

        # save optimizer for generator only
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
        # save generator
        torch.save({'a': self.gen_a.state_dict()}, gen_name)


####################################
# Baseline : qDL 2D
####################################
class qDL2D_Trainer(nn.Module):
    def __init__(self, hyperparameters, input_dim, output_dim):
        super(qDL2D_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = q_DL2D(input_dim, output_dim)  # auto-encoder for domain a

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen_a.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        gpu_ids = hyperparameters['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu') # get device name: CPU or GPU
        print('Deploy to {}'.format(self.device))
        self.gen_a.to(self.device)
        # Network weight initialization
        self.gen_a.apply(weights_init('orthogonal'))
        print('Init generator with {}'.format(hyperparameters['init']))

        self.loss_derives = torch.zeros([]).to(self.device)
        self.l1_w = hyperparameters['l1_w']

    def recon_criterion(self, input, target):
        brain_mask = 1.-(target==0.).float()
        pix_loss =  F.l1_loss(input, target, reduction='sum')
        pix_loss = pix_loss / (brain_mask.sum() + 1e-10)
        loss = pix_loss
        return loss

    def forward(self, b0):
        self.eval()
        x_img = self.gen_a.forward(b0)
        self.train()
        return x_img

    def sample(self, data_dict, trgs):
        self.gen_a.eval()
        return_dict = {}
        in_i = data_dict['dwi'].to(self.device).float()
        pred_i = self.gen_a.forward(in_i)
        target = None
        for ti in range(len(trgs)):
            return_dict['trg_%s' % trgs[ti]] = data_dict[trgs[ti]][0, 0].cpu().numpy()
            if target is None:
                target = data_dict[trgs[ti]].to(self.device).float()
            else:
                target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        loss_derives = self.recon_criterion(pred_i, target)
        self.gen_a.train()
        return_dict['dwi'] = in_i[0, 0].cpu().numpy()
        for ti in range(len(trgs)):
            return_dict[trgs[ti]] = pred_i[0,ti].detach().cpu().numpy()
        return_dict['loss'] = loss_derives.item()
        return return_dict


    def update(self, data_dict, epochs, trgs):
        self.gen_opt.zero_grad()
        in_i = data_dict['dwi'].to(self.device).float()
        target = None
        return_dict = {}
        for ti in range(len(trgs)):
            return_dict['trg_%s'%trgs[ti]] = data_dict[trgs[ti]][0,0].cpu().numpy()
            if target is None: target = data_dict[trgs[ti]].to(self.device).float()
            else: target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        pred_i = self.gen_a.forward(in_i)
        self.loss_derives = self.l1_w * self.recon_criterion(pred_i, target)
        self.loss_derives.backward()
        self.gen_opt.step()
        return_dict['dwi'] = in_i[0, 0].cpu().numpy()
        for ti in range(len(trgs)):
            return_dict['pred_%s'%trgs[ti]] = pred_i[0, ti].detach().cpu().numpy()
            print(trgs[ti], return_dict['pred_%s'%trgs[ti]].min(),return_dict['pred_%s'%trgs[ti]].max(), return_dict['trg_%s'%trgs[ti]].min(), return_dict['trg_%s'%trgs[ti]].max())
        return return_dict

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        print('Load models from %s'%checkpoint_dir)
        state_dict = torch.load(checkpoint_dir)
        self.gen_a.load_state_dict(state_dict['a'])
        # Load optimizers
        last_opt_name = checkpoint_dir.replace("gen", "opt")
        state_dict = torch.load(last_opt_name)
        self.gen_opt.load_state_dict(state_dict['gen'])
        return


    def save(self, snapshot_dir, epoch, step=-1, gen_name=None):
        # Save generators, discriminators, and optimizers
        if gen_name is None:
            if epoch == -1:
                gen_name = os.path.join(snapshot_dir, 'gen_latest.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_latest.pt')
            else:
                if step == -1:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%d.pt' % (epoch + 1))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%d.pt' % (epoch + 1))
                else:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%dstep%d.pt' % (epoch + 1, step))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%dstep%d.pt' % (epoch + 1, step))
        else:
            opt_name = gen_name.replace('gen','opt')

        # save optimizer for generator only
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
        # save generator
        torch.save({'a': self.gen_a.state_dict()}, gen_name)



