import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as _mp
from resnet3d import *
import queue


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, device, thread_limit=1):
        super(Generator, self).__init__()

        self.res = resnet10(sample_size=1, sample_duration=1)
        self.res.share_memory()
        self.mp = _mp.get_context('spawn')

        self.device = device
        self.batch_size = 50000
        self.thread_limit = thread_limit

    def proc_forward(self, X, Y, Z):
        #self.out[:, :, X, Y, Z] = self.res(self.x[:, :, X-1:X+2, Y-1:Y+2, Z-1:Z+2])
        print('hello ')

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        x = torch.cat([x, c], dim=1)


        # patches = torch.zeros([x.shape[0]*x.shape[2]*x.shape[3]*x.shape[4], x.shape[1], 3, 3, 3])
        x = torch.nn.functional.pad(x, (1, 1, 1, 1, 1, 1, 0, 0, 0, 0))
        processes = []

        voxel = 0

        in_q = self.mp.Queue()
        out_q = self.mp.Queue()
        for X in range(1, x.shape[2]-1):
            for Y in range(1, x.shape[3]-1):
                for Z in range(1, x.shape[4]-1):
                    # patches[voxel*x.shape[0]:voxel*x.shape[0]+x.shape[0], :, :, :, :] = x[:, :, X-1:X+2, Y-1:Y+2, Z-1:Z+2]
                    # voxel += 1

                    #p = self.mp.Process(target=self.proc_forward, args=(X,Y,Z))
                    in_q.put([x[:, :, X-1:X+2, Y-1:Y+2, Z-1:Z+2], X, Y, Z])
                    p = self.mp.Process(target=self.res,
                                        args=(in_q, out_q))
                    p.start()
                    processes.append(p)

                    if len(processes) >= self.thread_limit:
                        for p in processes:
                            p.join()
                            result = out_q.get()
                            pass

                        processes = []

                    # self.res(self.x[:, :, X - 1:X + 2, Y - 1:Y + 2, Z - 1:Z + 2])

        for p in processes:
            p.join()

        del self.x
        out = self.out
        del self.out

        return out

        # voxels = torch.Tensor().to(self.device)
        # while patches.shape[0] > 0:
        #     voxels = torch.cat((voxels, self.res(patches[0:min(self.batch_size,patches.shape[0])].to(self.device))), dim=0)
        #     patches = patches[min(self.batch_size,patches.shape[0]):]
        #
        # x = x[:, 0:15, 1:x.shape[2]-1, 1:x.shape[3]-1, 1:x.shape[4]-1]
        # for X in range(x.shape[2]):
        #     for Y in range(x.shape[3]):
        #         for Z in range(x.shape[4]):
        #             x[:, :, X, Y, Z] = voxels[0:x.shape[0],:]
        #             voxels = voxels[x.shape[0]:]
        # return x



class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv3d(15, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))