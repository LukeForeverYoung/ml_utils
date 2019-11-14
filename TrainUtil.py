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

