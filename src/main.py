from src.student_teacher.options import OptionsInformation
from src.utils.nn_utils import select_device
from student_teacher.train_student import create_student, load_teacher, load_model

standard_options = OptionsInformation(
    student_model="mnist_fc_256_128_10",
    device=select_device()
)


def main():
    #standard_options.reload_model_path = "saved_models/0.8681434392929077_f1Score__8_Epoch__2023_11_21_16_07_15_options.json_Options.pt"
    #load_model(standard_options, standard_options.reload_model_path)

    teacher_path = "C:\\Code\\KIT\\vnncomp2022_benchmarks\\benchmarks\\mnist_fc\\onnx\\mnist-net_256x6.onnx"
    teacher_model = load_teacher(teacher_path)
    student_model = create_student(teacher_model, standard_options)
    pass


if __name__ == "__main__":
    main()