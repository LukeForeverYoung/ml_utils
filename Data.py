<<<<<<< HEAD
from random import shuffle
def split_dataset(data,train_rate=0.8):
    shuffle(data)
    dl=len(data)
    return {
        'train':data[:int(dl*train_rate)],
        'valid':data[int(dl*train_rate):int(dl*(train_rate+1)/2)],
        'test':data[int(dl*(train_rate+1)/2):],
    }
class BatchDataLoader():
    def __init__(self,data,batch_size):
        self.__data=list(data)
        self.__batch_size=batch_size
        assert batch_size<=len(self.__data)
        self.__reset()

    def __reset(self):
        from random import shuffle
        self.__batch_pos=0
        self.__stop_flag=False
        shuffle(self.__data)

    def __iter__(self):
        return self

    def __next__(self):
        from random import sample
        if self.__stop_flag==True:
            self.__reset()
            raise StopIteration
        self.__batch_pos+=1
        if self.__batch_pos*self.__batch_size<=len(self.__data):
            return self.__data[(self.__batch_pos-1)*self.__batch_size:self.__batch_pos*self.__batch_size]
        else:
            tmp=self.__data[(self.__batch_pos-1)*self.__batch_size:]
            lake_num=self.__batch_pos*self.__batch_size-len(self.__data)
            tmp.extend(sample(self.__data,lake_num))
            self.__stop_flag=True
            return tmp

if __name__=='__main__':
    data=[i for i in range(100)]
    loader=BatchDataLoader(data,32)
    for batch in loader:
        print(batch)
    for batch in loader:
        print(batch)

=======
from random import shuffle

def move_to(obj, device):
    # 自动递归地将嵌套dict list的obj转化到cuda上
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k:move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    else:
        raise TypeError("Invalid type for move_to")
        
def split_dataset(data,train_rate=0.8):
    shuffle(data)
    dl=len(data)
    return {
        'train':data[:int(dl*train_rate)],
        'valid':data[int(dl*train_rate):int(dl*(train_rate+1)/2)],
        'test':data[int(dl*(train_rate+1)/2):],
    }
class BatchDataLoader():
    def __init__(self,data,batch_size):
        self.__data=list(data)
        self.__batch_size=batch_size
        assert batch_size<=len(self.__data)
        self.__reset()

    def __reset(self):
        from random import shuffle
        self.__batch_pos=0
        self.__stop_flag=False
        shuffle(self.__data)

    def __iter__(self):
        return self

    def __next__(self):
        from random import sample
        if self.__stop_flag==True:
            self.__reset()
            raise StopIteration
        self.__batch_pos+=1
        if self.__batch_pos*self.__batch_size<=len(self.__data):
            return self.__data[(self.__batch_pos-1)*self.__batch_size:self.__batch_pos*self.__batch_size]
        else:
            tmp=self.__data[(self.__batch_pos-1)*self.__batch_size:]
            lake_num=self.__batch_pos*self.__batch_size-len(self.__data)
            tmp.extend(sample(self.__data,lake_num))
            self.__stop_flag=True
            return tmp

if __name__=='__main__':
    data=[i for i in range(100)]
    loader=BatchDataLoader(data,32)
    for batch in loader:
        print(batch)
    for batch in loader:
        print(batch)

>>>>>>> 7dc76a8a26b4beeb9d4b68154905b1061b99c2d3
