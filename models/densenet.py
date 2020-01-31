class DenseNetZrc(torch.nn.Module):
    """
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each dense block
        num_init_featuremaps (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_featuremaps=64, bn_size=4, drop_rate=0, num_classes=1000, grayscale=False):
        super(DenseNetZrc, self).__init__()
        
        if grayscale:
            in_channels = 1
        else:
            in_channels = 3
            
        self.features = torch.nn.Sequential(OrderedDict([
            ( "conv_0", torch.nn.Conv2d(in_channels = in_channels,
                                       out_channels = num_init_featuremaps,
                                       kernel_size = 7,
                                       stride = 2,
                                       padding = 3, 
                                       bias = False) ),
            ( 'norm0',torch.nn.BatchNorm2d(num_features = num_init_featuremaps) ),
            ( 'relu0', torch.nn.ReLU(inplace = True) ),
            ( 'pool0', torch.nn.MaxPool2d(kernel_size = 3, 
                                          stride = 2, 
                                          padding = 1) )
        ]))  # [64, 7, 7] 
        
        
        num_features = num_init_featuremaps

        
        for index, num_layers in enumerate(block_config):
            
            dense_block = DenseBlock(num_layers = num_layers, 
                               num_input_features = num_features, 
                               bn_size = bn_size, 
                               growth_rate = growth_rate, 
                               drop_rate = drop_rate)
            
            self.features.add_module(
                "dense_block_{}".format(index+1), dense_block)
            num_features = num_features + num_layers * growth_rate
            
            if index != len(block_config) - 1:
                transition = Transition(num_input_features = num_features,
                                        num_output_features = num_features // 2)
                self.features.add_module(
                    "transition_{}".format(index+1), transition)
                
                num_features = num_features // 2
        
        
        # Final Batch Norm
        self.features.add_module("norm5",torch.nn.BatchNorm2d(num_features))
        self.classifier = torch.nn.Linear(num_features, num_classes)
        
        #############################################
        # initialize the weight
        #############################################
        for name,module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        probas = F.softmax(logits, dim=1)
        
        return logits, probas


class Transition(torch.nn.Module):
    """
    To bring down previous channel count
    """
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.composite = composite(input_features = num_input_features,
                                   out_features = num_output_features, 
                                   kernel_size = 1, 
                                   stride = 1, 
                                   padding = 0)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(2,2),
                                          stride = (2,2))
        
    def forward(self, x):
        return self.avg_pool(self.composite(x))


class DenseBlock(torch.nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        
        self.layers = torch.nn.ModuleList([
            DenseLayer(
            num_input_features + i * growth_rate,
            growth_rate = growth_rate,
            bn_size = bn_size,
            drop_rate = drop_rate,
            ) for i in range(num_layers)        
        ])
    
    def forward(self, x):
        x = [x]
        for layer in self.layers:
            x.append(layer(*x))
        
#         print(torch.cat(x,1).size()) # 64 + 6*32 = 256
        return torch.cat(x,1)

class DenseLayer(torch.nn.Module):
    def __init__(self, num_input_features, growth_rate = 32, bn_size = 2, drop_rate = 0):
        super(DenseLayer, self).__init__()
        
        self.composite_1 = composite(input_features = num_input_features,
                                     out_features = bn_size * growth_rate,
                                     kernel_size = 1,
                                     stride = 1,
                                     padding = 0)
        
        
        
        self.composite_2 = composite(input_features = bn_size * growth_rate,
                                     out_features = growth_rate,
                                     kernel_size = 3,
                                     stride = 1,
                                     padding = 1)
        
        self.drop_rate = drop_rate
            
    def forward(self, *prev_features):
        
        ## concate in channels
        concated_features = torch.cat(prev_features, 1)        
        bottleneck_output = self.composite_1(concated_features)
        new_features = self.composite_2(bottleneck_output)
        
        if self.drop_rate:
            new_features = F.dropout(new_features, p=self.drop_rate, training= self.training)
            
        return new_features

def composite(input_features,out_features, kernel_size, stride = 1, padding = 0):
    
    return torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(input_features, 
                            out_features, 
                            kernel_size=kernel_size, 
                            stride=stride,
                            padding = padding,
                            bias=False))