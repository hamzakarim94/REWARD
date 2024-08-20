import torch
from torch import nn

class CompressorCNN(nn.Module):
    def __init__(self):
        super(CompressorCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, dilation = 1, stride = 1, padding = 1),
                                   nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-05),
                                   nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, dilation = 2, stride = 1, padding = 2),
                                   nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-05),
                                   nn.ReLU()
                                   )
        self.conv3 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3, dilation = 4, stride = 1, padding = 4),
                                   nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-05),
                                   nn.ReLU()
                                   )
        self.conv4 = nn.Sequential(nn.Conv3d(256, 64, kernel_size=3,  stride = 1, padding = 1),
                                   nn.ReLU()
                                   )
        self.conv5 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(num_groups=8, num_channels=256, eps=1e-05),
                                   nn.ReLU()
                                   )
        self.attn  =importance_classifier(in_channels=64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.decoder_conv1 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.decoder_conv2 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x1 =  self.conv1(x)
        #x1 = self.dropout(x1)
        x2 =self.conv2(x)
        #x2 = self.dropout(x2)
        x3 = self.conv3(x)
        #x3 = self.dropout(x3)
        x_d = torch.cat((x1, x2, x3), dim=1)
        x4 =  self.conv4(x)
        x4 = self.attn(x4)
        x4 = torch.cat((x_d, x4), dim=1)
        x4 = self.conv5(x4)
        return x4+x

class importance_classifier(nn.Module):
    def __init__(self,in_channels):
        super(importance_classifier, self).__init__()
        self.conv_key = nn.Conv3d(in_channels, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv_query = nn.Conv3d(in_channels, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv_value = nn.Conv3d(in_channels, 32, kernel_size=3,stride = 1,  padding=1)
        self.alter = nn.Conv3d(32, 64, kernel_size=3, stride = 1, padding=1)
        #self.attn  = attention()
        #self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.decoder_conv1 = nn.ConvTranspose3d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.m = nn.GELU()
        #self.fc1 = nn.Linear(1568,512)
        #self.fc2 = nn.Linear(512,1568)
        #self.decoder_conv2 = nn.ConvTranspose3d(256, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        key=  self.conv_key(x)
        query = self.conv_query(x)
        value = self.conv_value(x)
        # Flatten the last three dimensions to get shape (8, 64, 196)
        encoder_shape = key.shape
        key = key.view(key.size(0),key.size(1), -1)
        query = query.view(query.size(0),query.size(1) ,-1)
        value = value.view(value.size(0),value.size(1), -1)
        #key = torch.relu(self.fc1(key))
        #query = torch.relu(self.fc1(query))
        #value = torch.relu(self.fc1(value))
        key = torch.permute(key, (1,0,2,))#(1,2,0,)(1,0,)(1,0,2,)
        value = torch.permute(value,(1,2,0) )#(1,0,2)(1,0)(1,2,0)
        query = torch.permute(query, (1, 2, 0))#(1, 0, 2)(1, 2, 0)
        attn = torch.softmax((query @ key),dim=1)
        out = torch.bmm(attn, value)
        #out = torch.permute(out,(2,0,1))
        #out = torch.relu(self.fc2(out))
        out = out.view(encoder_shape)
        out = self.alter(out)
        #out = self.decoder_conv2(out)
        return x+out

class VideoEncoder_AGG(nn.Module):
    def __init__(self):
        super(VideoEncoder_AGG, self).__init__()
        # Define the 3D convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=4, padding=1)  # Change 64 to 3
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, padding=1)
        #self.conv4 = nn.Conv3d(256, 512, kernel_size=4, padding=1)
        #self.conv5 = nn.Conv3d(512, 1024, kernel_size=4, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose3d(256, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                                                  output_padding=(0, 0, 0))
        self.conv_transpose3 = nn.ConvTranspose3d(128, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                                                  output_padding=(0, 0, 0))
        self.conv_transpose4 = nn.ConvTranspose3d(64, 3, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                                                  output_padding=(0, 0, 0))

        #self.aggregate = CompressorCNN()
        self.dropout = nn.Dropout3d(p=0.1)
        #self.pixel_shuffle = nn.PixelShuffle(8)
        #self.pixel_shuffle = UpscaleModel()
        # Create an instance of the self-attention module
        #self.attention_module = SelfAttention(in_channels=256)
        self.feat_agg1 = CompressorCNN()
        #self.feat_agg2 = TwoD_feature_agg(in_channels=784,dilation=1,padding=1)
        #self.feat_agg3 = TwoD_feature_agg(in_channels=784,dilation=2,padding=2)
    def forward(self, x):
        #print(x.shape)
        #x4 = torch.reshape(x,(x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
        #x4 = torch.permute(x4,(1,0,2,3))
        #x4 = self.aggregate(x4)
        #x4 = torch.reshape(x4,(1,x4.shape[0],1,1,1))
        x1 = torch.relu(self.pool(self.conv1(x)))
        #x1 = self.feat_agg1(x1)
        #x1= self.dropout(x1)
        #print(x1.shape)
        x2 = torch.relu(self.pool(self.conv2(x1)))
        #x2 = self.feat_agg2(x2)
        #x2 = self.dropout(x2)
        #print(x2.shape)
        x3 = torch.relu(self.pool(self.conv3(x2)))

        #encoder_shape = x3.shape
        x3= self.feat_agg1(x3)#torch.permute(x3,(0,2,3,4,1))
        #x3 = self.dropout(x3)
        #x3 = torch.permute(x3,(0,4,1,2,3))[0]

        #x3 = x3.view(encoder_shape)
        x = self.conv_transpose2(x3)
        #x = self.feat_agg2(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.conv_transpose3(x+x2)

        x = self.dropout(x)
        #print(x.shape)
        #x = self.pixel_shuffle(x)
        x = self.conv_transpose4(x+x1)

        #x = torch.permute(x,(0,2,1,3,4))*x4
        #print(x.shape)'''
        x = x.clamp(-0.225/2, 0.225/2)
        #x = torch.permute(x,(0,2,1,3,4))
        return x


class Image_attack(nn.Module):
    def __init__(self):
        super(Image_attack, self).__init__()
        # Define the 3D convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, padding=1)  # Change 64 to 3
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        #self.conv4 = nn.Conv3d(256, 512, kernel_size=4, padding=1)
        #self.conv5 = nn.Conv3d(512, 1024, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                                  output_padding=(0, 0))
        self.conv_transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                                  output_padding=(0, 0))
        self.conv_transpose4 = nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                                  output_padding=(0, 0))

        #self.aggregate = CompressorCNN()
        self.dropout = nn.Dropout3d(p=0.1)
        #self.pixel_shuffle = nn.PixelShuffle(8)
        #self.pixel_shuffle = UpscaleModel()
        # Create an instance of the self-attention module
        #self.attention_module = SelfAttention(in_channels=256)
        #self.feat_agg1 = CompressorCNN()
        #self.feat_agg2 = TwoD_feature_agg(in_channels=784,dilation=1,padding=1)
        #self.feat_agg3 = TwoD_feature_agg(in_channels=784,dilation=2,padding=2)
    def forward(self, x):
        #print(x.shape)
        pre_size = x.size()
        x = x.permute(1, 0, 2, 3, 4).reshape(1,3, x.shape[0]*x.shape[2], 224, 224)
        x = torch.permute(x, (1, 0, 2, 3, 4))
        shape = x.size()
        x= torch.reshape(x, (shape[0],  shape[2], shape[3], shape[4]))

        x = torch.permute(x,(1,0,2,3))
        #x4 = torch.reshape(x,(x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
        #x4 = torch.permute(x4,(1,0,2,3))
        #x4 = self.aggregate(x4)
        #x4 = torch.reshape(x4,(1,x4.shape[0],1,1,1))
        x1 = self.pool(torch.relu(self.conv1(x)))
        #x1 = self.feat_agg1(x1)
        #x1= self.dropout(x1)
        #print(x1.shape)
        x2 = self.pool(torch.relu(self.conv2(x1)))
        #x2 = self.feat_agg2(x2)
        #x2 = self.dropout(x2)
        #print(x2.shape)
        x3 = self.pool(torch.relu(self.conv3(x2)))

        #encoder_shape = x3.shape
        #x3= self.feat_agg1(x3)#torch.permute(x3,(0,2,3,4,1))
        #x3 = self.dropout(x3)
        #x3 = torch.permute(x3,(0,4,1,2,3))[0]

        #x3 = x3.view(encoder_shape)
        x = self.conv_transpose2(x3)
        #x = self.feat_agg2(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.conv_transpose3(x+x2)
        #x = self.feat_agg1(x)
        x = self.dropout(x)
        #print(x.shape)
        #x = self.pixel_shuffle(x)
        x = self.conv_transpose4(x+x1)

        x = torch.permute(x,(1,0,2,3))
        #print(x.shape)'''
        x = x.clamp(-0.225/2, 0.225/2)
        x = x.reshape(pre_size[1],pre_size[0],pre_size[2],pre_size[3],pre_size[4]).permute(1, 0, 2, 3, 4)

        #x = torch.permute(x,(0,2,1,3,4))
        return x



class VideoEncoder_NOAGG(nn.Module):
    def __init__(self):
        super(VideoEncoder_NOAGG, self).__init__()
        # Define the 3D convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=4, padding=1)  # Change 64 to 3
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, padding=1)
        #self.conv4 = nn.Conv3d(256, 512, kernel_size=4, padding=1)
        #self.conv5 = nn.Conv3d(512, 1024, kernel_size=4, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose3d(256, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                                                  output_padding=(0, 0, 0))
        self.conv_transpose3 = nn.ConvTranspose3d(128, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                                                  output_padding=(0, 0, 0))
        self.conv_transpose4 = nn.ConvTranspose3d(64, 3, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                                                  output_padding=(0, 0, 0))

        #self.aggregate = CompressorCNN()
        self.dropout = nn.Dropout3d(p=0.1)
        #self.pixel_shuffle = nn.PixelShuffle(8)
        #self.pixel_shuffle = UpscaleModel()
        # Create an instance of the self-attention module
        #self.attention_module = SelfAttention(in_channels=256)
        #self.feat_agg1 = CompressorCNN()
        #self.feat_agg2 = TwoD_feature_agg(in_channels=784,dilation=1,padding=1)
        #self.feat_agg3 = TwoD_feature_agg(in_channels=784,dilation=2,padding=2)
    def forward(self, x):
        #print(x.shape)
        #x4 = torch.reshape(x,(x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
        #x4 = torch.permute(x4,(1,0,2,3))
        #x4 = self.aggregate(x4)
        #x4 = torch.reshape(x4,(1,x4.shape[0],1,1,1))
        x1 = torch.relu(self.pool(self.conv1(x)))
        #x1 = self.feat_agg1(x1)
        #x1= self.dropout(x1)
        #print(x1.shape)
        x2 = torch.relu(self.pool(self.conv2(x1)))
        #x2 = self.feat_agg2(x2)
        #x2 = self.dropout(x2)
        #print(x2.shape)
        x3 = torch.relu(self.pool(self.conv3(x2)))

        #encoder_shape = x3.shape
        #x3= self.feat_agg1(x3)#torch.permute(x3,(0,2,3,4,1))
        #x3 = self.dropout(x3)
        #x3 = torch.permute(x3,(0,4,1,2,3))[0]

        #x3 = x3.view(encoder_shape)
        x = self.conv_transpose2(x3)
        #x = self.feat_agg2(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.conv_transpose3(x+x2)

        x = self.dropout(x)
        #print(x.shape)
        #x = self.pixel_shuffle(x)
        x = self.conv_transpose4(x+x1)

        #x = torch.permute(x,(0,2,1,3,4))*x4
        #print(x.shape)'''
        x = x.clamp(-0.225/2, 0.225/2)
        #x = torch.permute(x,(0,2,1,3,4))
        return x

