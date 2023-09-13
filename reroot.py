import argparse
from phylo_utils import select_root, reroot


def args_parser():
    parser = argparse.ArgumentParser(description="BCR-SORT")
    parser.add_argument("--file_phylo", "--i",
                        help="Input fasta file containing codon-aligned sequence to construct lineage", type=str)
    parser.add_argument("--file_pred", "--r",
                        help="Input csv file containing cell subset information to select new root", type=str)
    parser.add_argument("--igphyml_path", "--p",
                        help="Path of IgPhyML", type=str)

    return parser.parse_args()


def main():
    args = args_parser()
    root_id = select_root(file_in=args.file_pred)
    reroot(args, root_id)


if __name__ == "__main__":
    main()
