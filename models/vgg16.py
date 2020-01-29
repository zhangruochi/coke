class Vgg16Zrc(torch.nn.Module):
    def __init__(self, num_classes, grascale = False):
        super(Vgg16Zrc, self).__init__()
        if grascale:
            in_channels = 1
        else:
            in_channels = 3
            
        #[3*224*224] -> [64*224*224]
        self.block_1 = torch.nn.Sequential(
        
            torch.nn.Conv2d(in_channels = in_channels, 
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            
            torch.nn.Conv2d(64, 
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
        )
        
        
        #[64*224*224] -> [128*112*112]
        self.block_2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size = 2,
                              stride = 2),
            torch.nn.Conv2d(in_channels = 64, 
                            out_channels = 128,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            
            torch.nn.Conv2d(128, 
                            out_channels = 128,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
        )
        
        # [128*112*112] -> [256*56*56]
        self.block_3 = self.__block_helper(128,256)
        # [256*56*56] -> [512*28*28]
        self.block_4 = self.__block_helper(256,512)
        # [512*28*28] -> [512*14*14]
        self.block_5 = self.__block_helper(512,512)
        
        
        self.classifier = torch.nn.Sequential(
            # [512*14*14] -> [512*7*7]
            torch.nn.MaxPool2d(kernel_size = 2,
                              stride = 2),
#             torch.nn.AdaptiveAvgPool2d((7,7)),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            
            # [512*7*7] -> [4096]
            torch.nn.Linear(in_features = 512, out_features = 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            
            # [4096] -> [4096]
            torch.nn.Linear(in_features = 4096, out_features = 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.5),
            
            # [4096] -> [num_classes]
            torch.nn.Linear(in_features = 4096, out_features = num_classes)
        )
        
        self.layers = torch.nn.ModuleList([self.block_1,self.block_2,self.block_3,self.block_4,self.block_5])
        
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits,probas
    
    
    
    def __block_helper(self,in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size = 2,
                              stride = 2),
            torch.nn.Conv2d(in_channels = in_channels, 
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(),
            
            torch.nn.Conv2d(in_channels = out_channels, 
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(),
            
            torch.nn.Conv2d(in_channels= out_channels, 
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout()
            
        )