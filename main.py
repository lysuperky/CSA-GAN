from __future__ import print_function
import multiprocessing

import os
import io
import sys
import time
import errno
import random
import pprint
import datetime
import dateutil.tz
import argparse
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import clever_format
from thop import profile
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchstat import stat

from miscc.utils import mkdir_p
from miscc.utils import imagenet_deprocess_batch
from miscc.config import cfg, cfg_from_file
from miscc.losses import DAMSM_loss
from sync_batchnorm import DataParallelWithCallback
#from datasets_everycap import TextDataset
from datasets import TextDataset
from datasets import prepare_data
from DAMSM import RNN_ENCODER, CNN_ENCODER
from model import NetG, NetD
from model import CA
from model import DCM

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 200


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='6')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, ca_net, dataloader, ixtoword, device):
    model_dir = cfg.TRAIN.NET_G
    istart = cfg.TRAIN.NET_G.rfind('_') + 1
    iend = cfg.TRAIN.NET_G.rfind('.')
    start_epoch = int(cfg.TRAIN.NET_G[istart:iend])

    '''
    for path_count in range(11):
        if path_count > 0:
            current_epoch = next_epoch
        else:
            current_epoch = start_epoch
        next_epoch = start_epoch + path_count * 10
        model_dir = model_dir.replace(str(current_epoch), str(next_epoch))
    '''
    # hard debug by setting the index of trained epoch, adjust it as your need

    split_dir = 'valid'
    #split_dir = 'test_every'
    # Build and load the generator
    netG.load_state_dict(torch.load(model_dir,map_location= "cuda:0"),strict = False)
    netG.eval()
    # ca_net.eavl()

    batch_size = cfg.TRAIN.BATCH_SIZE
    #s_tmp = model_dir
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    s_tmp_dir = s_tmp
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    real_img_save_dir = '%s/%s' % (s_tmp, 'real')
    mkdir_p(real_img_save_dir)
    cap_save_dir = '%s/%s' % (s_tmp, 'caps')
    mkdir_p(cap_save_dir)

    idx = 0
    cnt = 0
    # hard debug by specifyng the number of synthezied images from caption
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            # imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            imags, captions, cap_lens, class_ids, keys, wrong_caps, wrong_caps_len, wrong_cls_id = prepare_data(data)

            real_imgs = imags[0].to(device)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            # c_code, mu, logvar = ca_net(sent_emb) 
            cnn_code, region_features ,mask = None, None, None
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            # fake, _ = netG(noise, sent_emb, cnn_code, region_features, c_code,mask)

            # code for generating captions
            # cap_imgs = cap2img_new(ixtoword, captions, cap_lens, s_tmp_dir)
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise = noise.to(device)
                c_code, mu, logvar = ca_net(sent_emb) 
                fake_imgs, stage_masks = netG(noise, sent_emb, c_code)
                stage_mask = stage_masks[-1]
            for j in range(batch_size):
                # save generated image
                s_tmp = '%s/img' % (fake_img_save_dir)
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)

                #fullpath = '%s_%3d.png' % (s_tmp,i)
                fullpath = '%s_s%d.png' % (s_tmp, idx)
                im.save(fullpath)

                # save the last fusion mask
                s_tmp = '%s/fm' % fake_img_save_dir
                im = stage_mask[j].data.cpu().numpy()
                # [0, 1] --> [0, 255]
                # im = 1-im # only for better visualization
                im = im * 255.0
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = np.squeeze(im, axis=2)
                im = Image.fromarray(im)
                fullpath = '%s_%d.png' % (s_tmp, idx)
                im.save(fullpath)

                idx += 1
# def load_model_weights(model, weights, multi_gpus, train=True):
#     # if list(weights.keys())[0].find('module')==-1:
#     pretrained_with_multi_gpu = False
#     # else:
#     #     pretrained_with_multi_gpu = True
#     if (multi_gpus==False) or (train==False):
#         if pretrained_with_multi_gpu:
#             state_dict = {
#                 key[7:]: value
#                 for key, value in weights.items()
#             }
#         else:
#             state_dict = weights
#     else:
#         state_dict = weights
#     model.load_state_dict(state_dict)
#     return model


def gen_sample(text_encoder, netG, device, wordtoix):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """
    # captions = ['A colorful blue bird has wings with dark stripes and small eyes',
    #             'A colorful green bird has wings with dark stripes and small eyes',
    #             'A colorful white bird has wings with dark stripes and small eyes',
    #             'A colorful black bird has wings with dark stripes and small eyes',
    #             'A colorful pink bird has wings with dark stripes and small eyes',
    #             'A colorful orange bird has wings with dark stripes and small eyes',
    #             'A colorful brown bird has wings with dark stripes and small eyes',
    #             'A colorful red bird has wings with dark stripes and small eyes',
    #             'A colorful yellow bird has wings with dark stripes and small eyes',
    #             'A colorful purple bird has wings with dark stripes and small eyes']

    # captions = ['A herd of black and white cattle standing on a field',
    #  'A herd of black cattle standing on a field',
    #  'A herd of white cattle standing on a field',
    #  'A herd of brown cattle standing on a field',
    #  'A herd of black and white sheep standing on a field',
    #  'A herd of black sheep standing on a field',
    #  'A herd of white sheep standing on a field',
    #  'A herd of brown sheep standing on a field']
    # captions = ['A herd of black and white sheep standing on a field']

    captions = [
    'some dogs',
    ]
    # captions = ['A group of cars on a street with a domed building in the background']
    # captions = ['some horses in a field of green grass with a sky in the background']
    #  'some horses in a field of yellow grass with a sky in the background',
    #  'some horses in a field of green grass with a sunset in the background',
    #  'some horses in a field of yellow grass with a sunset in the background']

    # caption to idx
    # split string to word
    for c, i in enumerate(captions):
        captions[c] = i.split()

    caps = torch.zeros((len(captions), 18), dtype=torch.int64)

    for cl, line in enumerate(captions):
        for cw, word in enumerate(line):
            caps[cl][cw] = wordtoix[word.lower()]
    caps = caps.to(device)
    cap_len = []
    for i in captions:
        cap_len.append(len(i))

    caps_lens = torch.tensor(cap_len, dtype=torch.int64).to(device)
    model_dir = cfg.TRAIN.NET_G
    # netG = load_model_weights(netG,model_dir,False)
    netG.load_state_dict(torch.load(model_dir,map_location='cuda:0'))
    # from 
    # 
    split_dir = 'valid'
    # new_state_dic=OrderedDict()
    # for k,v in model_state_dict.items():
    #     name=k[7:]
    #     new_state_dic[name]=v
    # model.load_state_dict(new_state_dict)
    # model=model.cuda()
    netG.eval()
    netG.to(device)
    # ca_net.val()

    batch_size = len(captions)
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    for step in range(1):

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(caps, caps_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        # c_code, mu, logvar = ca_net(sent_emb)
        #######################################################
        # (2) Generate fake images
        ######################################################
        with torch.no_grad():
            noise = torch.randn(1, 100) # using fixed noise
            noise = noise.repeat(batch_size, 1)
            # use different noise
            noise = []
            for i in range(batch_size):
                noise.append(torch.randn(1, 100))
            noise = torch.cat(noise,0)
            
            noise = noise.to(device)

            # noise = noise.to(device)
            # noise = torch.nn.functional.normalize((noise - mu) /logvar)
            c_code, mu, logvar = ca_net(sent_emb) 
            # print(noise.cpu(), sent_emb, c_code)
            fake_imgs, stage_masks = netG(noise.to(device), sent_emb.to(device), c_code.to(device))
            # input = torch.randn(1, 3, 224, 224)
            # model = netG.cpu()
            # import os 
            # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )            
            # flops, params = profile(netG.to(device), inputs=(noise.to(device), sent_emb.to(device), c_code.to(device)))
            # flops, params = clever_format([flops, params], "%.3f")
            # stat(netG, [3,100,100])
            # print('flops: ', flops, 'params: ', params)
            # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
            stage_mask = stage_masks[-1]
        for j in range(batch_size):
            # save generated image
            s_tmp = '%s/img' % fake_img_save_dir
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            # print(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            # fullpath = '%s_%3d.png' % (s_tmp,i)
            fullpath = '%s_%d.png' % (s_tmp, step)
            print(fullpath)
            im.save(fullpath)

            # save fusion mask
            s_tmp = '%s/fm' % fake_img_save_dir
            im = stage_mask[j].data.cpu().numpy()
            # im = 1-im # only for better visualization
            # [0, 1] --> [0, 255]
            im = im * 255.0
            im = im.astype(np.uint8)

            im = np.transpose(im, (1, 2, 0))
            im = np.squeeze(im, axis=2)
            im = Image.fromarray(im)
            fullpath = '%s_%d.png' % (s_tmp, step)
            print(fullpath)
            im.save(fullpath)


def cap2img(ixtoword, caps, cap_lens):
    imgs = []
    for cap, cap_len in zip(caps, cap_lens):
        idx = cap[:cap_len].cpu().numpy()
        caption = []
        for i, index in enumerate(idx, start=1):
            caption.append(ixtoword[index])
            if i % 4 == 0 and i > 0:
                caption.append("\n")
        caption = " ".join(caption)
        fig = plt.figure(figsize=(2.5, 1.5))
        plt.axis("off")
        plt.text(0.5, 0.5, caption)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    assert imgs.dim() == 4, "image dimension must be 4D"
    return imgs
# def cap2img_new(ixtoword, caps, cap_lens,s_tmp_dir):
#     imgs = []
#     for cap, cap_len in zip(caps, cap_lens):
#         idx = cap[:cap_len].cpu().numpy()
#         caption = []
#         for i, index in enumerate(idx, start=1):
#             caption.append(ixtoword[index])
#             if i % 4 == 0 and i > 0:
#                 caption.append("\n")
#         caption = " ".join(caption)
#         fig = plt.figure(figsize=(2.5, 1.5))
#         plt.axis("off")
#         plt.text(0.5, 0.5, caption)
#         plt.xlim(0, 10)
#         plt.ylim(0, 10)
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         plt.close(fig)
#         buf.seek(0)
#         img = Image.open(buf).convert('RGB')
#         img = transforms.ToTensor()(img)
#         imgs.append(img)
#     imgs = torch.stack(imgs, dim=0)
#     assert imgs.dim() == 4, "image dimension must be 4D"
    
#     return imgs

def write_images_losses(writer, imgs, fake_imgs, errD, d_loss, errG, DAMSM, epoch, cap_imgs):
    index = epoch
    writer.add_scalar('errD/d_loss', errD, index)
    writer.add_scalar('errD/MAGP', d_loss, index)
    writer.add_scalar('errG/g_loss', errG, index)
    writer.add_scalar('errG/DAMSM', DAMSM, index)
    imgs_print = imagenet_deprocess_batch(imgs)
    cap_imgs =  imagenet_deprocess_batch(cap_imgs)
    #imgs_64_print = imagenet_deprocess_batch(fake_imgs[0])
    #imgs_128_print = imagenet_deprocess_batch(fake_imgs[1])
    imgs_256_print = imagenet_deprocess_batch(fake_imgs)
    writer.add_image('images/img1_pred', torchvision.utils.make_grid(imgs_256_print, normalize=True, scale_each=True), index)
    writer.add_image('images/img2_caption', torchvision.utils.make_grid(cap_imgs, normalize=True, scale_each=True), index)
    writer.add_image('images/img3_real', torchvision.utils.make_grid(imgs_print, normalize=True, scale_each=True), index)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def prepare_labels(batch_size):
    # Kai: real_labels and fake_labels have data type: torch.float32
    # match_labels has data type: torch.int64
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()
    return real_labels, fake_labels, match_labels


def train(dataloader, ixtoword, netG, netD, ca_net, text_encoder, image_encoder,
          optimizerG, optimizerD, state_epoch, batch_size, device):
    base_dir = os.path.join('tmp', cfg.CONFIG_NAME, str(cfg.TRAIN.NF))

    if not cfg.RESTORE:
        writer = SummaryWriter(os.path.join(base_dir, 'writer'))
    else:
        writer = SummaryWriter(os.path.join(base_dir, 'writer_new'))

    mkdir_p('%s/models' % base_dir)
    real_labels, fake_labels, match_labels = prepare_labels(batch_size)

    # Build and load the generator and discriminator
    if cfg.RESTORE:
        model_dir = cfg.TRAIN.NET_G
        netG.load_state_dict(torch.load(model_dir))
        model_dir_D = model_dir.replace('netG', 'netD')
        netD.load_state_dict(torch.load(model_dir_D))
        netG.train()
        netD.train()
        ca_net.train()
        # dcm.train()
        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        state_epoch = int(cfg.TRAIN.NET_G[istart:iend])

    for epoch in tqdm(range(state_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1)):
        data_iter = iter(dataloader)
        # for step, data in enumerate(dataloader, 0):
        for step in tqdm(range(len(data_iter))):
            data = data_iter.next()

            # imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            imgs, captions, cap_lens, class_ids, keys, wrong_caps, wrong_caps_len, wrong_cls_id = prepare_data(data)
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            c_code, mu, logvar = ca_net(sent_emb)
            region_features, cnn_code = image_encoder(imgs[0])

            
            # w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
            # w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

            imgs = imgs[0].to(device)
            real_features = netD(imgs)
            output = netD.module.COND_DNET(real_features, sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.module.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()
            noise = torch.randn(batch_size, 100)
            noise = noise.to(device)

            # _, _, _, _, h_code, c_code = g(noise, sent_emb, words_embs, mask, cnn_code, real_features)

            # mask = (captions == 0)
            # num_words = words_embs.size(2)
            # if mask.size(1) > num_words:
            #     mask = mask[:, :num_words]
            # mask = mask.to(device)
            # synthesize fake images

            # fake_img = dcm(h_code, real_feat    ures, sent_emb, words_embs,\
            #                     mask, c_code)
            # fake, _ = netG(noise, sent_emb)
            # fake, _ = netG(noise, sent_emb, cnn_code, region_features, c_code,mask)
            fake, _ = netG(noise, sent_emb, c_code)

            # G does not need update with D
            fake_features = netD(fake.detach())
            errD_fake = netD.module.COND_DNET(fake_features, sent_emb)
            
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            errD = errD_real + (errD_fake + errD_mismatch) / 2.0
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()

            # MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.module.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # update G
            features = netD(fake)
            output = netD.module.COND_DNET(features, sent_emb)
            errG = - output.mean()
            DAMSM = 0.05 * DAMSM_loss(image_encoder, fake, real_labels, words_embs,
                                      sent_emb, match_labels, cap_lens, class_ids)
            errG_total = errG + DAMSM
            optimizerG.zero_grad()
            errG_total.backward()
            optimizerG.step()

        # caption can be converted to image and shown in tensorboard
        cap_imgs = cap2img(ixtoword, captions, cap_lens)

        write_images_losses(writer, imgs, fake, errD, d_loss, errG, DAMSM, epoch,cap_imgs)
        if epoch < 100:
            if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) or (epoch % cfg.TRAIN.GSAVE_INTERVAL == 0):
                torch.save(netG.state_dict(), '%s/models/netG_%03d.pth' % (base_dir, epoch))
            if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) or (epoch % cfg.TRAIN.DSAVE_INTERVAL == 0):
                torch.save(netD.state_dict(), '%s/models/netD_%03d.pth' % (base_dir, epoch))
        else:
            if epoch % cfg.TRAIN.GSAVE_INTERVAL == 0:
                torch.save(netG.state_dict(), '%s/models/netG_%03d.pth' % (base_dir, epoch))
            if epoch % cfg.TRAIN.DSAVE_INTERVAL == 0:
                torch.save(netD.state_dict(), '%s/models/netD_%03d.pth' % (base_dir, epoch))


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id
    
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 3407
    elif args.manualSeed is None:
        args.manualSeed = 3407
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    # torch.cuda.set_device(cfg.GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    cudnn.benchmark = True
    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        wordtoix = dataset.wordtoix
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            drop_last=True,
            shuffle=True, 
            num_workers=int(cfg.WORKERS)
        )

    # # validation data #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # alpha = Variable(torch.ones(1,1), requires_grad = True).to(device)

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    netG = DataParallelWithCallback(netG,device_ids=[0])
    netD = nn.DataParallel(netD)
    ca_net = CA().to(device)
    # dcm = DCM().to(device)
    # g = G_NET().to(device)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder.cuda()
    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    image_encoder.eval()

    state_epoch = 0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    if cfg.B_VALIDATION:
        # sampling(text_encoder, netG, ca_net,dataloader, ixtoword, device)  # generate images for the whole valid dataset
        gen_sample(text_encoder, netG, device, wordtoix) # generate images with description from user
    else:
        train(dataloader, ixtoword, netG, netD, ca_net,text_encoder, image_encoder, optimizerG, optimizerD, state_epoch, batch_size, device)
