import time
class TimeCoster():

    def __init__(self):
        self.__time_tick=time.time()
        self.__time_start=time.time()
    def start(self):
        self.__time_tick = time.time()

    def tock(self,print_s=False):
        time_cost=time.time() - self.__time_tick
        self.__time_tick=time.time()
        if print_s:
            self.print_time(time_cost)
        return time_cost

    def cost_from_start(self,print_s=False):
        time_cost=time.time() - self.__time_start
        if print_s:
            self.print_time(time_cost)
        return time_cost

    def print_time(self,s):
        print('cost {:.2g} s'.format(s))

class TrainUtil():
    def __init__(self,print_limit=100):
        self.print_limit=print_limit
        self.clear()

    def clear(self):
        self.epoch=0
        self.loss_sum=0

    def add_loss(self,v):
        self.loss_sum+=v
        self.epoch+=1
        if self.epoch%self.print_limit==0:
            self.print_loss()
            self.loss_sum=0

    def print_loss(self):
        print('ep:{0} loss:{1}'.format(self.epoch, self.loss_sum))
if __name__ == '__main__':

    train_hp=TrainUtil(10)
    for i in range(100):
        train_hp.add_loss(10)
    input()
    coster=TimeCoster()
    time.sleep(5)
    coster.tock(print_s=True)
    time.sleep(5)
    coster.tock(print_s=True)
    coster.cost_from_start(print_s=True)
