import torch

class OptimizerUtil():
    def __init__(self,lr,optimizer):
        self.lr=lr
        self.opt=optimizer

    def adjust(self,epoch=None):
        if epoch:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = self.lr * (0.1 ** (epoch // 30))
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.lr * 0.1
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr

class Trainer():
    def __init__(self,model,loaders,optimizer,save_path,stop_mode='noincrease',threshold=None,cuda_convertor=None,logger=None,device='cuda'):
        self.model=model
        self.train_loader=loaders['train'] if 'train' in loaders else None
        self.valid_loader=loaders['valid'] if 'valid' in loaders else None
        self.test_loader=loaders['test'] if 'test' in loaders else None
        self.optimizer=optimizer
        self.stop_mode=stop_mode
        self.threshold=threshold
        self.logger=logger
        self.device=device
        self.save_path=save_path
        self.cuda_convertor=cuda_convertor
        if self.device:
            self.model.to(self.device)
    def train(self,):
        info={}
        info['best_ep']=0
        info['best_score']=0
        if self.stop_mode is not 'noincrease':
            for ep in range(self.threshold):
                info['ep']=ep
                info=self.train_body(info)
        else:
            ep=0
            info['stop_cnt']=self.threshold
            while True:
                info['ep']=ep
                info=self.train_body(info)
                if info['stop_cnt']==0:
                    break
                ep+=1
    def train_body(self,info):
        self.model.train()
        for bi,(data,target) in enumerate(self.train_loader):
            if self.cuda:
                if self.cuda_convertor:
                    data=self.cuda_convertor(data)
                    target=self.cuda_convertor(target)
                else:
                    data, target = data.to(self.device), target.to(self.device)
            
        vinfo=self.valid()
        if vinfo['score']>info['best_score']:
            info['best_score']=vinfo['score']
            info['best_ep']=info['ep']
            if self.stop_mode=='noincrease':
                info['stop_cnt']-=1
        else:
            if self.stop_mode=='noincrease':
                info['stop_cnt']=self.threshold
            self.save()
    
    def save(self,):
        with open(self.save_path,'wb')as f:
            torch.save(self.model.state_dict(), f)
    def load(self,path):
        if type(path) is str:
            self.model.load_state_dict(torch.load(path))
        else:
            with open(path,'rb')as f:
                self.model.load_state_dict(torch.load(f))

