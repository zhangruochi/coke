class NiNZrc(torch.nn.Module):
    def __init__(self, num_classes, grayscale = False):
        super(NiNZrc, self).__init__()
        
        if grayscale:
            in_channels = 1
        else:
            in_channels = 3
            
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 192, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(0.5),

            torch.nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(0.5),

            torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192,  num_classes, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()
            
        
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.classifier(x)
        logits = torch.squeeze(self.global_avg_pooling(x))
        probas = torch.softmax(logits, dim=1)
        return logits, probas