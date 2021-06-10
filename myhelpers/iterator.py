
class Infinite_iter():
    def __init__(self, loader):
        self.iter = iter(loader)
        self.loader = loader
        self.counter = 0
        # print('len_iter', len(self.iter))

    def getBatch(self):
        try:
            # print('hi', len(self.iter), len(self.loader))
            batch = next(self.iter) 
        except StopIteration:
            self.counter = 0
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            self.iter = iter(self.loader)
            batch = next(self.iter) 
            # print('ho', len(self.iter), len(self.loader))

        # print('iter counter', self.counter)
        self.counter = self.counter + 1
        # print(batch)
        return batch