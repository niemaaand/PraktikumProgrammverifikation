from src.debug.build_debug_network import build_debug_network
from src.student_teacher.options import OptionsInformation, read_options_from_json, DatasetType
from src.utils.nn_utils import select_device
from student_teacher.train_student import create_student, load_teacher, load_model


no_teacher_options = OptionsInformation(
    n_epochs=20,
    student_model="mnist_fc_784_256_256_10",
    vnnlibs_path="",
    #vnnlibs_path="",
    #teacher_path="C:\\Code\\KIT\\vnncomp2022_benchmarks\\benchmarks\\mnist_fc\\onnx\\mnist-net_256x2.onnx",
    teacher_path="",
    dataset=DatasetType.MNIST,
    #criterion="BCEWithLogitsLoss",
    #criterion="MSELoss",
    criterion="CrossEntropyLoss",
    soft_target_loss_weight=0.0,
    ce_loss_weight=1.0,
    T=20,
    adapt_T=True,
    force_save=True,
    adapt_T_steps=2,
    choose_best_model_based_on_teacher=False,
    device=select_device()
)


mnist_options = OptionsInformation(
    n_epochs=20,
    student_model="mnist_fc_784_32_10",
    vnnlibs_path="",
    #vnnlibs_path="",
    #teacher_path="C:\\Code\\KIT\\vnncomp2022_benchmarks\\benchmarks\\mnist_fc\\onnx\\mnist-net_256x2.onnx",
    teacher_path="onnx/2024_02_20_10_46_42_options.json_Options__0.9854999780654907_f1Score__19_Epoch.onnx",
    dataset=DatasetType.MNIST,
    #criterion="BCEWithLogitsLoss",
    criterion="MSELoss",
    #criterion="CrossEntropyLoss",
    soft_target_loss_weight=0.25,
    ce_loss_weight=0.75,
    T=20,
    adapt_T=True,
    force_save=True,
    adapt_T_steps=2,
    choose_best_model_based_on_teacher=True,
    device=select_device()
)

acasxu_options = OptionsInformation(
    student_model="acasxu_0",
    vnnlibs_path="C:/Code/KIT/vnncomp2022_benchmarks/benchmarks/acasxu/vnnlib/prop_1.vnnlib",
    dataset=DatasetType.RANDOM_INPUTS,
    teacher_path="C:\\Code\\KIT\\vnncomp2022_benchmarks\\benchmarks\\acasxu\\onnx\\ACASXU_run2a_1_1_batch_2000.onnx",
    device=select_device()
)


standard_options = mnist_options


def main():
    #options = read_options_from_json("saved_models/2024_01_13_15_47_33_options.json")
    #options.reload_model_path = "saved_models/2024_01_13_15_47_33_options.json_Options__0.8756420612335205_f1Score__0_Epoch.pt"
    #load_model(standard_options, options.reload_model_path)

    #options = OptionsInformation(student_model="mini_debug_model2_BIG")
    #build_debug_network("C:/Code/KIT/Praktikum_Programmverifikation/examples/debug/mini2BIG.onnx", options)

    for i in range(3):
        teacher_path = standard_options.teacher_path
        teacher_model = load_teacher(teacher_path)
        student_model = create_student(teacher_model, standard_options)
    pass


if __name__ == "__main__":
    main()