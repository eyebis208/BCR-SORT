import argparse
from model_utils import predict


def args_parser():
    parser = argparse.ArgumentParser(description="BCR-SORT")
    parser.add_argument("--input_file", "--i",
                        help="Input file to predict cell subsets", type=str)
    parser.add_argument("--output_file", "--o", default=None,
                        help="Output file containing prediction results of cell subsets", type=str)
    parser.add_argument("--model_path", "--p", default='./model_weight/model_wt.pt',
                        help="Path of model weights", type=str)
    parser.add_argument("--device", default=0,
                        help="GPU for prediction", type=int)

    return parser.parse_args()


def main():
    args = args_parser()
    predict(args)


if __name__ == "__main__":
    main()
