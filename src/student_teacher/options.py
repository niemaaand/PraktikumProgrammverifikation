import enum
import json
import os


class DatasetType(enum.Enum):
    RANDOM_INPUTS = 0
    MNIST = 1

    def __mydict__(self):
        return {
            "_name_": self.name,
            "_value_": self.value
        }


class OptionsInformation:
    def __init__(self,
                 batch_size=128,
                 n_batches=200,
                 n_batches_validation=10,
                 criterion="CrossEntropyLoss",
                 device="cpu",
                 lr=0.005,
                 lr_momentum=0,
                 n_epochs=10,
                 optimizer="adam", #"sgd",
                 reload_model_path="",
                 scheduler="StepLR",
                 scheduler_lr_gamma=0.85,
                 scheduler_step=2,
                 #student_model="mnist_fc_256_128_10",
                 student_model="acasxu_0",
                 teacher_path="C:\\Code\\KIT\\vnncomp2022_benchmarks\\benchmarks\\acasxu\\onnx\\ACASXU_run2a_1_1_batch_2000.onnx",
                 saving_threshold=0.8,
                 dataset=DatasetType.RANDOM_INPUTS,
                 average_type_in_metrics="micro",
                 #vnnlibs_path="C:/Code/KIT/vnncomp2022_benchmarks/benchmarks/mnist_fc/vnnlib",
                 vnnlibs_path="C:/Code/KIT/vnncomp2022_benchmarks/benchmarks/acasxu/vnnlib",
                 soft_target_loss_weight=0.25,
                 ce_loss_weight=0.75,
                 T=2,
                 adapt_T=False,
                 adapt_T_steps=1,
                 force_save=False,
                 choose_best_model_based_on_teacher=False):

        self.batch_size = batch_size
        self.n_batches = n_batches  # for training with random inputs
        self.n_batches_validation = n_batches_validation  # for validation with random inputs
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
        self.teacher_path = teacher_path
        self.saving_threshold = saving_threshold
        self.dataset = dataset
        self.average_type_in_metrics = average_type_in_metrics
        self.vnnlibs_path = vnnlibs_path  # only relevant with random input samples
        self.soft_target_loss_weight = soft_target_loss_weight  # soft_target_loss_weight and ce_loss_weight have to add up to 1
        self.ce_loss_weight = ce_loss_weight  # soft_target_loss_weight and ce_loss_weight have to add up to 1
        self.T = T
        self.adapt_T = adapt_T
        self.adapt_T_steps = adapt_T_steps
        self.force_save = force_save
        self.choose_best_model_based_on_teacher = choose_best_model_based_on_teacher # if False, then best student is chosen based on labels of data

    def write_options_to_json(self, path: str):
        dict_opt = vars(self)

        with open(path, "w") as f:
            json.dump(dict_opt, f, default=lambda o: o.__dict__ if type(o) != DatasetType else o.__mydict__(), indent=4)


def read_options_from_json(path: str) -> OptionsInformation:
    file = open(path, "r")
    dict_options = json.load(file)
    json_s = json.dumps(dict_options)
    o = OptionsInformation(**json.loads(json_s))
    return o
