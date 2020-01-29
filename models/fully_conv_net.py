class FullyConvNetZrc(torch.nn.Module):
    def __init__(self, num_classes):
        super(FullyConvNetZrc, self).__init__()
        
        # (w - k + 2*p)/s + 1
        
        # [1*28*28] -> [4*28*28]
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=4,
                                     kernel_size=(3,3),
                                     stride=(1,1),
                                     padding=1)
        
        
        # [4*28*28] -> [4*14*14]
        self.conv_2 = torch.nn.Conv2d(in_channels=4,
                                     out_channels=4,
                                     kernel_size=(4,4),
                                     stride=(2,2),
                                     padding=1)
        
        # [4*14*14] -> [8*14*14]
        self.conv_3 = torch.nn.Conv2d(in_channels=4,
                                     out_channels=8,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        # [8*14*14] -> [8*7*7]
        self.conv_4 = torch.nn.Conv2d(in_channels=8,
                                     out_channels=8,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1)
        # [8*7*7] -> [16*7*7]
        self.conv_5 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)
        
        # [16*7*7] -> [16*4*4]
        self.conv_6 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        
        # [16*4*4] -> [num_classes*4*4]
        self.conv_7 = torch.nn.Conv2d(in_channels=16,
                                      out_channels= num_classes,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)
        
        # [num_classes*4*4] -> [num_classes]
        self.out_pool = torch.nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        
        out = self.conv_2(out)
        out = F.relu(out)

        out = self.conv_3(out)
        out = F.relu(out)

        out = self.conv_4(out)
        out = F.relu(out)
        
        out = self.conv_5(out)
        out = F.relu(out)
        
        out = self.conv_6(out)
        out = F.relu(out)
        
        out = self.conv_7(out)
        out = F.relu(out)
        
        logits = torch.squeeze(self.out_pool(out))
        
        probas = torch.softmax(logits, dim = 1)
        
        return logits,probas