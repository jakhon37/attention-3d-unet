class InitParser(object):
    def __init__(self):
        # gpu setting
        self.gpu_id = 0

        # optimizer setting
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-4

        # train setting
        self.batch_size = 1
        self.train_ratio = 0.8
        self.num_epoch = 15
        self.init_epoch = 1
        self.is_load = False

        # path setting
        self.output_path = ( r"C:\Users\jakho\PycharmProjects\projects\tutorial\3dunt_simple\output\UNet3D")
        self.data_path = (r"C:\Users\jakho\PycharmProjects\projects\tutorial\3dunt_simple\data\Original")

        # self.data_path = (r"E:\data\brats2020\train\no_patch")

        # E:\data\brats2020\train\no_patch

        self.load_path = (r"C:\Users\jakho\PycharmProjects\projects\tutorial\3dunt_simple\output\UNet3D\Network_{}.pth.gz".format(self.init_epoch - 1))