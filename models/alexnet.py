class AlextNetZrc(torch.nn.Module):

    def __init__(self, num_classes, grascale = False):
        
        super(AlextNetZrc, self).__init__()
        
        if grascale:
            in_channels = 1
        else:
            in_channels = 3
            
        self.block_1 = torch.nn.Sequential(
    
            torch.nn.Conv2d(in_channels = in_channels,
                            out_channels = 96,
                            kernel_size = 11,
                            stride = 4,
                            padding = 4),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            torch.nn.BatchNorm2d(96),
            
            torch.nn.Conv2d(in_channels = 96,
                            out_channels = 256,
                            kernel_size = 5,
                            padding = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            torch.nn.BatchNorm2d(256),
        
        )
        
        
        self.block_2 = torch.nn.Sequential(*[self.__block_2_helper(in_channels,out_channels) for in_channels,out_channels in [(256,384),(384, 384),(384,256)]])
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256*6*6,4096),
            torch.nn.ReLU(inplace = True),
            
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(inplace = True),
            
            torch.nn.Linear(4096,num_classes)
        )
        
    def forward(self,x):
        x = self.block_1(x)
#         print(x.size())
        x = self.block_2(x)
#         print(x.size())
        x = self.max_pool(x)
        
        x = torch.flatten(x,start_dim=1, end_dim=-1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
        
        
        
    def __block_2_helper(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size = 3,padding = 1),
            torch.nn.ReLU(inplace = True)
        )