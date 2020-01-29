class ConvNetZrc(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNetZrc, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p) / s + 1 = w   =>  p = ( s( w - 1) - w + k) / 2
        
        # [1*28*28] => [4*28*28]
        # 28
        self.conv_1 = torch.nn.Conv2d(in_channels = 1,
                                    out_channels = 4,
                                    kernel_size = (3,3),
                                    stride = (1, 1),
                                    padding = 1) # (1(28-1) - 28 + 3) / 2 = 1
        
        # [4*28*28] => [4*14*14]
        self.pool_1 = torch.nn.MaxPool2d(kernel_size = (2,2),
                                        stride = (2,2),
                                        padding = 0) 
        
        # [4*14*14] => [8*14*14]
        self.conv_2 = torch.nn.Conv2d(in_channels = 4, 
                                     out_channels = 8,
                                     kernel_size = (3,3),
                                     stride = (1, 1),
                                     padding = 1)
        
        # [8*14*14] => [8*7*7]
        self.pool_2 = torch.nn.MaxPool2d(kernel_size = (2,2),
                                        stride = (2,2),
                                        padding = 0)
        
        self.fully_connected_layer = torch.nn.Linear(8*7*7, num_classes)
        
    
    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.pool_1(x)
        
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool_2(x)
        
        logits = self.fully_connected_layer(x.view(-1, 8*7*7))
        probas = F.softmax(logits, dim = 1)
        
        return logits,probas