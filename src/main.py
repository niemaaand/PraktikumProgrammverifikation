from src.student_teacher.options import OptionsInformation
from src.utils.nn_utils import select_device
from student_teacher.train_student import create_student, load_teacher

standard_options = OptionsInformation(
    device=select_device()
)


def main():
    teacher_path = "C:\\Code\\KIT\\vnncomp2022_benchmarks\\benchmarks\\mnist_fc\\onnx\\mnist-net_256x6.onnx"
    teacher_model = load_teacher(teacher_path)
    student_model = create_student(teacher_model, standard_options)
    pass

if __name__ == "__main__":
    main()