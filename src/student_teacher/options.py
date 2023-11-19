class OptionsInformation:
    def __init__(self,
                 batch_size=128,
                 criterion="CrossEntropyLoss",
                 device="cpu",
                 lr=0.03,
                 lr_momentum=0,
                 n_epochs=20,
                 optimizer="sgd",
                 reload_model_path="",
                 scheduler="StepLR",
                 scheduler_lr_gamma=0.85,
                 scheduler_step=2,
                 student_model="mnist_fc_256_128_10"):

        self.batch_size = batch_size
        self.criterion = criterion
        self.device = device
        self.lr = lr
        self.lr_momentum = lr_momentum
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.reload_model_path = reload_model_path
        self.scheduler = scheduler
        self.scheduler_lr_gamma = scheduler_lr_gamma
        self.scheduler_step = scheduler_step
        self.student_model = student_model
