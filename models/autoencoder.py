class AutoEncoderZrc(torch.nn.Module):
    def __init__(self, grayscale = False):
        super(AutoEncoderZrc, self).__init__()
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        if grayscale:
            in_channels = 1
        else:
            in_channels = 3
        
        
        # (w-k+2p) // 2 + 1
        
        # 28x28x1 => 14x14x4
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels=4,
                            kernel_size=(3, 3),
                            stride=(2, 2),
                            padding=1),
            torch.nn.LeakyReLU(inplace = True),
            # 14x14x4 => 7x7x8
            torch.nn.Conv2d(in_channels=4,
                              out_channels=8,
                              kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=1),
            torch.nn.LeakyReLU(inplace = True)
        )
        
        # Hout=(H−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=8,
                                     out_channels=4,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding = 1,
                                     output_padding = 1),
            torch.nn.LeakyReLU(inplace = True),
            torch.nn.ConvTranspose2d(in_channels = 4,
                            out_channels= in_channels,
                            kernel_size=(3, 3),
                            stride=(2, 2),
                            padding = 1,
                            output_padding = 1),
                            
            torch.nn.LeakyReLU(inplace = True)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x