class  VariationalAutoencoderZrc(torch.nn.Module):
    def __init__(self, num_latent, grayscale = False):
        super(VariationalAutoencoderZrc, self).__init__()
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
            torch.nn.LeakyReLU(inplace = True),
            torch.nn.Flatten(),
            
        )
        
        self.z_mean = torch.nn.Linear(8*8*8, num_latent)
        self.z_log_var = torch.nn.Linear(8*8*8, num_latent)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use 
        # an exponential function
        
        # Hout=(H−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 8*8*8),
            Reshape((-1,8,8,8)),
            
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
        
    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(DEVICE)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
        
    def forward(self, x):
        x = self.encoder(x)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        
        encoded = self.reparameterize(mean, log_var)
        decoded = torch.sigmoid(self.decoder(encoded))
        
        
        return mean,log_var,decoded