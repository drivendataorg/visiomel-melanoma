# Third-party libraries
import torch.nn as nn
import torchvision
import torch

__all__ = ["BreslowUlcerationRelapseModel", "ImageTabularModel", "DummyModel"]

class BreslowUlcerationRelapseModel(nn.Module):
    def __init__(self):
        super(BreslowUlcerationRelapseModel, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.breslow = nn.Linear(512, 5)
        self.ulceration = nn.Linear(512, 1)
        self.relapse = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        # breslow = nn.Softmax()(self.breslow(x))
        breslow = self.breslow(x)
        ulceration = self.sigmoid(self.ulceration(x))
        relapse = self.sigmoid(self.relapse(x))
        return breslow, ulceration, relapse
    

class DummyModel(nn.Module):
    def __init__(self, model_path):
        super(DummyModel, self).__init__()
        pretrained = BreslowUlcerationRelapseModel()
        pretrained.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))


        self.image_model_freeze = pretrained.resnet
        self.image_model_freeze.fc = nn.Identity()

    def forward(self, x):
        x = self.image_model_freeze(x)
        return x

class ImageTabularModel(nn.Module):
    def __init__(self, tabular_inps, model_type, dropout=0.35, relapse_only=True):
        super(ImageTabularModel, self).__init__()

        assert model_type in ["FC", "CNN"], f"Model type should be 'FC' or 'CNN' "


        self.tabular_inps = tabular_inps
        self.model_type = model_type
        self.dropout = dropout
        self.relapse_only = relapse_only
        if model_type == "CNN":
            self.image_model = nn.Sequential(
                                nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(),

                                nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),

                                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),

                                nn.Conv2d(64, 8, kernel_size=3, padding=1, stride=1),
                                nn.BatchNorm2d(8),
                                nn.ReLU(),

                                nn.MaxPool2d(2, stride=2),
                                )
        else:
            self.image_model = nn.Sequential(
                                nn.Linear(512,1024),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),

                                nn.Linear(1024,1024),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                
                                nn.Linear(1024,128),
                                nn.ReLU(),
                                nn.Dropout(self.dropout))

        self.tabular_model = nn.Sequential(
                            nn.Linear(tabular_inps,64),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),

                            nn.Linear(64,128),
                            nn.ReLU(),
                            nn.Dropout(self.dropout))
        if self.relapse_only:
            self.classifier = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout), 

                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),

                                nn.Linear(256, 1))
        else:
            self.classifier = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout), 

                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(self.dropout))
            self.breslow = nn.Linear(256, 5)
            self.ulceration = nn.Linear(256, 1)
            self.relapse = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_latent, tabular):
        img_latent = image_latent
        if self.model_type == "CNN":
            img_latent = img_latent.view(-1,8,8,8)
        image_features = self.image_model(img_latent)

        tabular_features = self.tabular_model(tabular)

        features = torch.cat((image_features.view(-1, 128), tabular_features.view(-1, 128)), 1)

        if self.relapse_only:
            out = self.classifier(features)
            out = self.sigmoid(out)
        else:
            features = self.classifier(features)
            breslow = self.breslow(features) #nn.Softmax()(self.breslow(features))
            ulceration = self.sigmoid(self.ulceration(features))
            relapse = self.sigmoid(self.relapse(features))
            
            out = (breslow, ulceration, relapse)
        return out