class ResNetZrc(torch.nn.Module):

    def __init__(self, block, layers, grasacale = False, num_classes=1000):
        super(ResNetZrc, self).__init__()
        
        self.inplanes = 64
        
        if grasacale:
            init_channels = 1
        else:
            init_channels = 3
        
        
        self.pre = torch.nn.Sequential(OrderedDict([
            ("conv0", torch.nn.Conv2d(init_channels, 
                                      self.inplanes, 
                                      kernel_size=7, 
                                      stride=2, 
                                      padding=3,
                                      bias=False)),
            ("bn0", torch.nn.BatchNorm2d(self.inplanes)),
            ("relu0", torch.nn.ReLU(inplace=True)),
            ("pool0", torch.nn.MaxPool2d(kernel_size = 3, stride=2, padding=1))
            
        ]))
        
        
        self.middle_layers = torch.nn.Sequential(
            self._make_layers(block, 64,  layers[0]),
            self._make_layers(block, 128, layers[1], stride=2),
            self._make_layers(block, 256, layers[2], stride=2),
            self._make_layers(block, 512, layers[3], stride=2)
        )
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)
        
        
        for name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()
            
    
    def _make_layers(self, block, planes, multiplier, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, multiplier):
            layers.append(block(self.inplanes, planes))
        
        return torch.nn.Sequential(*layers)    
    
    def forward(self, x):
        x = self.pre(x)
        x = self.middle_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x).squeeze()
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas



class Bottleneck(torch.nn.Module):
    expansion = 4
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()        
        self.seq = torch.nn.Sequential(
            # [1x1]
            torch.nn.Conv2d(inplanes, 
                            outplanes, 
                            kernel_size=1, 
                            bias=False),
            torch.nn.BatchNorm2d(outplanes),
            
            # [3x3]
            torch.nn.Conv2d(outplanes, 
                            outplanes, 
                            kernel_size=3, 
                            stride=stride,
                            padding=1, 
                            bias=False),
            torch.nn.BatchNorm2d(outplanes),
            
            # [1x1]
            torch.nn.Conv2d(outplanes, 
                                   outplanes * 4, 
                                   kernel_size=1, 
                                   bias=False),
            torch.nn.BatchNorm2d(outplanes * 4))
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.seq(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        return F.relu(out)

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, inplanes, 
                 outplanes, 
                 stride = 1, 
                 downsample = None):
        
        super(BasicBlock, self).__init__()
        
        self.seq = torch.nn.Sequential(
            conv3x3(inplanes,outplanes,stride),
            torch.nn.BatchNorm2d(outplanes),
            torch.nn.ReLU(inplace = True),
            
            
            conv3x3(outplanes,outplanes),
            torch.nn.BatchNorm2d(outplanes),      
        )
                
        self.downsample = downsample
        self.stride = stride
        
        
    def forward(self, x):
        residule = x
        
        out = self.seq(x)
        
        if self.downsample is not None:
            residule = self.downsample(x)
        
        out += residule
        return F.relu(out)


def conv3x3(inplanes, outplanes, stride = 1):
    return torch.nn.Conv2d(inplanes, 
                           outplanes, 
                           stride=stride,
                           kernel_size=3, 
                           padding = 1,
                           bias=False)