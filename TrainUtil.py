import torch
from collections import Iterable
from tqdm import tqdm
from os.path import join
import argparse
from torch import nn
import numpy as np
import random
import math
from pathlib import Path
import logging
from logging import getLogger,FileHandler,StreamHandler,Formatter
def set_seed(seed,cudnn_fixed=False):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs) 
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cudnn_fixed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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



def evaluator_accuracy_example(pred,ground):
    pred=(pred>0.5).long()
    return {'score':torch.mean((torch.eq(pred,ground)).float())}

class Initializer(object):

    @staticmethod
    def manual_seed(seed):
        """
        Set all of random seed to seed.
        --------------------
        Arguments:
                seed (int): seed number.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def xavier_normal(module, lstm_forget_bias_init=2):
        """
        Xavier Gaussian initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv1d, nn.Conv2d, nn.Linear)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.xavier_normal_(
                module.weight.data) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(
                    0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(
                        param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[hidden_size//4:hidden_size //
                                   2] = lstm_forget_bias_init

    @staticmethod
    def xavier_uniform(module, lstm_forget_bias_init=2):
        """
        Xavier Uniform initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.xavier_uniform_(
                module.weight.data) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(
                    0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_uniform_(
                        param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[hidden_size//4:hidden_size //
                                   2] = lstm_forget_bias_init

    @staticmethod
    def orthogonal(module, lstm_forget_bias_init=2):
        """
        Orthogonal initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.orthogonal_(
                module.weight.data) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(
                    0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(
                        param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[hidden_size//4:hidden_size //
                                   2] = lstm_forget_bias_init
    

class Trainer():
    def __init__(self,model,t_dataloader,v_dataloader,args):
        self.args=args
        self.runtime=argparse.Namespace()
        self.model=model
        self.model.cuda()
        self.t_dataloader=t_dataloader
        self.v_dataloader=v_dataloader
        self.loss_fn=None # 需要编写
        self.optimizer=None # 需要编写
        
        num_step_epoch=len(t_dataloader)
        self.lr_scheduler = None # 需要编写
       
        from time import strftime,localtime
        self.time_token=strftime("%Y-%m-%d[%H_%M_%S]", localtime())
        self.__set_logger()
        print(self.logger.handlers)


    def train(self,):
        self.runtime.epoch=0
        self.runtime.bad_epoch=0
        self.runtime.best_loss=math.inf
        valid_res={'valid_loss':math.inf}
        self.valid()
        while True:
            self.model.train()
            for bi,batch in enumerate(self.t_dataloader):
                # Train Loop
                '''
                batch={_:batch[_].cuda() for _ in batch if'exp_' not in _}
                inp={_:batch[_] for _ in batch if _!='ground_truth' and 'exp_' not in _}
                ground_truth=batch['ground_truth']
                output=self.model(**inp)
                loss=calculate_loss_and_accuracy(output,ground_truth)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
                self.optimizer.step()
                self.lr_scheduler.step()

                self.logger.info(f'ep:{self.runtime.epoch} {bi}/{len(self.t_dataloader)}, Loss:{loss.item():.4f}, Valid Loss:{valid_res["valid_loss"]:.4f}')
                '''
            valid_res=self.valid()
            if valid_res['valid_loss']<self.runtime.best_loss:
                self.runtime.bad_epoch=0
                self.runtime.best_loss=valid_res['valid_loss']
            else:
                self.runtime.bad_epoch+=1
            
            if self.runtime.bad_epoch>self.args.num_epoch:
                break
            #self.save(self.runtime.epoch)
            self.save() # 覆盖
            self.runtime.epoch+=1

    def valid(self,):
        self.model.eval()
        tot_loss=0

        with torch.no_grad():
            for bi,batch in enumerate(self.v_dataloader):
                # Valid Loop
                '''
                batch={_:batch[_].cuda() for _ in batch if'exp_' not in _}
                inp={_:batch[_] for _ in batch if _!='ground_truth' and 'exp_' not in _}
                ground_truth=batch['ground_truth']
                output=self.model(**inp)
                loss=calculate_loss_and_accuracy(output,ground_truth)
                tot_loss+=loss.item()
                '''

        return {'valid_loss':tot_loss}

    def save(self,epoch=None):
        suffix='checkpoint'
        if epoch is not None:
            suffix=f'checkpoint_{epoch}'
        
        output_dir=Path(self.args.output_dir,self.time_token,suffix)
        output_dir.mkdir(parents=True,exist_ok=True)
        
        self.logger.info("Saving model checkpoint to %s", output_dir)
        checkpoint={
            'args':self.args,
            'runtime':self.runtime,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
        }
        # 保存参数
        torch.save(checkpoint, Path(output_dir, "checkpoint.bin"))
    
    def load(self,path):
        checkpoint=torch.load(path)
        self.args=checkpoint['args']
        self.runtime=checkpoint['runtime']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __set_logger(self,):
        self.logger=getLogger('luke')
        self.logger.setLevel(logging.INFO) # windows下默认为warning, linux下默认为Info

        self.logger.propagate=False # 避免被transformers库中的logger配置污染
        formatter = Formatter('%(asctime)s [%(levelname)s] %(message)s',"%Y-%m-%d %H:%M:%S")
        log_dir=Path('log')
        log_dir.mkdir(parents=True,exist_ok=True)
        fh=FileHandler(Path(log_dir,f'{self.time_token}.log'))
        fh.setFormatter(formatter)
        sh=StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

if __name__=='__main__':
    a=torch.tensor([0.7,0.2,0.9,0.8])
    b=torch.tensor([1,1,0,0])
    print(evaluator_accuracy_example(a,b))
