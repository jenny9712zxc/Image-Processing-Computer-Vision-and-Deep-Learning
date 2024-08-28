from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
import numpy as np
import matplotlib.pyplot as plt
import window05_ui as ui
import train
from torchsummary import summary
from nn_module_vgg16 import VGG16
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as transforms

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  

        self.pushButton.clicked.connect(self.show9images)
        self.pushButton_2.clicked.connect(self.printHyperparameters)
        self.pushButton_4.clicked.connect(self.printModelShortcut)
        self.pushButton_3.clicked.connect(self.printAccurancy)
        self.pushButton_5.clicked.connect(self.test)

    def show9images(self):    
        # load_cifar_10_data
        data_dir = "data/cifar-10-batches-py"
    
        meta_data_dict = unpickle(data_dir+"/batches.meta")
        #meta_key    [b'num_cases_per_batch', b'label_names', b'num_vis']
        cifar_label_names = meta_data_dict[b'label_names']    
        cifar_label_names = np.array(cifar_label_names)
        #[b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

        # training data
        cifar_train_data = None
        cifar_train_filenames = []
        cifar_train_labels = []


        datadict = unpickle(data_dir+"/test_batch")
        #print(datadict.keys())#batch_key  [b'batch_label', b'labels', b'data', b'filenames']

        

        X = datadict[b'data'] 
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        Y = np.array(Y)   
        cifar_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        plt.ion()
        #Visualizing CIFAR 10
        fig, axes1 = plt.subplots(3,3,figsize=(5,5))
        for j in range(3):
            for k in range(3):
                i = np.random.choice(range(len(X)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[i:i+1][0]) 
                index = Y[i]
                index =  int(index)
                axes1[j][k].set_title(cifar_label_names[index])
        

    def printHyperparameters(self):
        #print("Batch Size {} ",  format(Batchsize variable name))
        print("hyperparameters:")
        print("batch size: " + str(train.args.batch_size))
        print("learing rate: " + str(train.args.lr))
        print("optimizer: SGD")

    def printModelShortcut(self):        
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")        
        net = VGG16().to(device)    
        summary(net, (3, 32, 32))

    def printAccurancy(self):
        plt.imshow(mpimg.imread('plot.png'))
        plt.show()

    def test(self):   
        mytext = self.textEdit.toPlainText()
        index = int(mytext)
        #print(index)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=False, num_workers=0)
        # get some random training images
        dataiter = iter(trainloader)
        for i in range(index):
            dataiter.next()
        images, labels = dataiter.next()


        #print(images.shape)#torch.Size([1, 3, 32, 32])
        device = torch.device('cpu')
        net = VGG16()
        net.load_state_dict(torch.load('pretrainedModel.pth', map_location=device))


        outputs = net(images )


        classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img = images.clone().detach()
        img = img.reshape(len(img), 3, 32, 32).permute(0,2,3,1)#.astype("uint8")


        plt.subplot(1, 2, 1)       
        plt.imshow(img[0].numpy())



        plt.subplot(1, 2, 2)
        x = np.arange(len(classes))

        out=outputs.detach().numpy()

        plt.bar(list(classes), out.flatten())
        plt.show()
        
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted] ))

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())