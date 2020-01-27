class SoftmaxRegression_zrc(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression_zrc, self).__init__()
        self.liner = torch.nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        logits = self.liner(x)
        probas = F.softmax(logits, dim = 1)
        return logits, probas
    
model = SoftmaxRegression_zrc(num_features= num_features, 
                              num_classes = num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)