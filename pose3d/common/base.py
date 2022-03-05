import os
import os.path as osp
import math
import time
import glob
import abc
import tensorflow as tf


from timer import Timer
from logger import colorlogger

from config import cfg

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.ckpt'.format(str(epoch)))
        # TODO
        tf.saved_model.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        # TODO
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])

        return start_epoch, model, optimizer


class Trainer(Base):
    
    def __init__(self, backbone):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')
        self.backbone = backbone

    def get_optimizer(self, model):
        # TODO
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            if i > 0:
                ref_joints_name = trainset3d_loader[0].joints_name
            else:
                ref_joints_name = None
            trainset3d_loader.append(DatasetLoader(eval(cfg.trainset_3d[i])("train"), ref_joints_name, True)
        ref_joints_name = trainset3d_loader[0].joints_name
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(DatasetLoader(eval(cfg.trainset_2d[i])("train"), ref_joints_name, True)

        self.joint_num = trainset3d_loader[0].joint_num

        trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_pose_net(self.backbone, True, self.joint_num)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
