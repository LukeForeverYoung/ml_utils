import torch
from collections import Iterable
from tqdm import tqdm
from os.path import join
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
    def __init__(self,model,loaders,optimizer,save_path,evaluator,stop_mode='noincrease',threshold=5,cuda_convertor=None,logger=None,device='cuda'):
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
        self.evaluator=evaluator
        self.cuda_convertor=cuda_convertor
        self.train_history=[]
        if self.device:
            self.model.to(self.device)
    
    def init(self,):
        if self.device is not 'cpu':
            self.model.to(self.device)
    
    def train(self,):
        info={}
        info['best_ep']=0
        info['best_score']=0
        self.train_history=[]
        if self.stop_mode is not 'noincrease':
            for ep in range(self.threshold):
                info['ep']=ep
                self.train_body(info)
        else:
            ep=0
            info['stop_cnt']=self.threshold
            while True:
                info['ep']=ep
                self.train_body(info)
                if info['stop_cnt']==0:
                    break
                ep+=1
        
        
    def train_body(self,info):
        self.model.train()
        tot_loss=0
        pred=[]
        ground=[]
        for bi,(data,target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data,target=self.cuda_data((data,target))
            res=self.model.forward(data)
            pred.append(torch.argmax(res,dim=1).detach().cpu())
            ground.append(target.detach().cpu())
            loss=self.model.loss(res,target)
            loss.backward()
            self.optimizer.step()
            tot_loss+=loss.item()
        pred=torch.cat(pred,dim=0)
        ground=torch.cat(ground,dim=0)
        info['loss']=tot_loss
        info['train_acc']=self.evaluator(pred,ground)['score']

        self.train_history.append({'loss':tot_loss})
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
    
    def valid(self,loader=None):
        self.model.eval()
        vinfo={}
        if loader is None:
            loader=self.valid_loader
        pred=[]
        ground=[]
        with torch.no_grad():
            for bi,(data,target) in enumerate(tqdm(loader)):
                data=self.cuda_data(data)
                res=self.model.tforward(data)
                pred.append(res.detach().cpu())
                ground.append(target.detach().cpu())
            pred=torch.cat(pred,dim=0)
            ground=torch.cat(ground,dim=0)

            vinfo=self.evaluator(pred,ground)
        if not isinstance(vinfo,dict):
            vinfo={'score':vinfo}
        
        return vinfo

    def cuda_data(self,data):
        if not torch.is_tensor(data):
            res=tuple([self.cuda_convertor(d) if self.cuda_convertor else d.to(self.device) for d in data])
            return res
        else:
            if self.cuda_convertor:
                return self.cuda_convertor(data)
            else:
                return data.to(self.device)


    def save(self,):
        with open(join(self.save_path,'best.pkl'),'wb')as f:
            torch.save(self.model.state_dict(), f)
    def load(self,path):
        if type(path) is str:
            self.model.load_state_dict(torch.load(path))
        else:
            with open(path,'rb')as f:
                self.model.load_state_dict(torch.load(f))


if __name__=='__main__':
    a=torch.tensor([0.7,0.2,0.9,0.8])
    b=torch.tensor([1,1,0,0])
    print(evaluator_accuracy_example(a,b))
