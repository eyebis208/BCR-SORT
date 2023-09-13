import argparse
from model_utils import train


def args_parser():
    parser = argparse.ArgumentParser(description="BCR-SORT")
    parser.add_argument("--input_file", "--i",
                        help="Input file to train the model", type=str)
    parser.add_argument("--output_dir", "--o", default=None,
                        help="Directory to save output files", type=str)
    parser.add_argument("--learning_rate", "--lr", default=1e-4,
                        help="Learning rate", type=float)
    parser.add_argument("--weight_decay", "--wd", default=0.05,
                        help="Weight decay", type=float)
    parser.add_argument("--loss_scaling", default=0.05,
                        help="Scaling ratio of auxiliary loss", type=float)
    parser.add_argument("--batch_size", default=1024,
                        help="Size of mini-batch", type=int)
    parser.add_argument("--num_epoch", default=10,
                        help="Number of epoch to train", type=int)
    parser.add_argument("--seq_dim", default=256,
                        help="Amino acid embedding dimension", type=int)
    parser.add_argument("--feature_dim", default=64,
                        help="Annotation feature embedding dimension", type=int)
    parser.add_argument("--hidden_dim", default=512,
                        help="LSTM hidden dimension", type=int)
    parser.add_argument("--layer_lstm", default=2,
                        help="Number of LSTM layer", type=int)
    parser.add_argument("--dropout_lstm", default=0.1,
                        help="Dropout rate of LSTM layer", type=float)
    parser.add_argument("--proj_lstm", default=0,
                        help="Projection of LSTM layer", type=int)
    parser.add_argument("--num_channel", default=64,
                        help="Number of CNN channel", type=int)
    parser.add_argument("--kernel_size", default=[3, 4, 5],
                        help="List of three kernel sizes", type=list)
    parser.add_argument("--stride", default=1,
                        help="Stride of the convolution", type=int)
    parser.add_argument("--dilation", default=1,
                        help="Spacing between kernels", type=int)
    parser.add_argument("--dropout_fc", default=0.5,
                        help="Dropout rate of FC layer", type=float)
    parser.add_argument("--dim_fc1", default=128,
                        help="Dimension of 1st FC layer", type=int)
    parser.add_argument("--dim_fc2", default=32,
                        help="Dimension of 2nd FC layer", type=int)
    parser.add_argument("--device", default=0,
                        help="GPU for prediction", type=int)

    return parser.parse_args()


def main():
    args = args_parser()
    train(args)


if __name__ == "__main__":
    main()
