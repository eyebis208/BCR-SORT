# BCR-SORT: Identification of B cell subsets based on antigen receptor sequences using deep learning

BCR-SORT is a deep learning model that predicts B cell subsets from their corresponding B cell receptor (BCR) sequences 

by leveraging B cell activation and maturation signatures encoded within BCR sequences, 

especially within the heavy chain amino acid sequence of complementarity-determining region 3 (CDR3).


# Predicting B cell subsets utilizing pretrained BCR-SORT

You can directly apply BCR-SORT on your BCR sequence datasets utilizing the pretrained BCR-SORT.

Input file must contain CDR3 amino acid sequence, IGHV gene usage, IGHJ gene usage, and isotype information as a .csv format.

(columns required: cdr3_aa, V_gene, J_gene, isotype)

```bash
predict.py --i <input file> --o <output file> 
```


# Training BCR-SORT on pre-defined datasets

Alternatively, you can train your own BCR-SORT using custom datasets.

Users can train the model with pre-defined hyperparameters by simply running the command below, or use their own values (see help function).

Input file must contain CDR3 amino acid sequence, IGHV gene usage, IGHJ gene usage, isotype, label information as a .csv format.

(columns required: cdr3_aa, V_gene, J_gene, isotype, label)

```bash
train.py --i <training data file> --o <output directory> 
```


# Cell subset-aware reconstruction of B cell lineage reconstruction

After running the BCR-SORT, predicted cell subset information can be further utilized in reconstructing the BCR phylogenetic tree.

To reconstruct a lineage following the biology of B cell differentiation, you can construct a phylogenetic tree rooted by the naive B cell 

(with the least mutations in the entire IGHV region, if it's given additionally) defined by BCR-SORT.

Input sequence file should be .fasta format containing codon-aligned BCR sequences constituting the phylogenetic tree, with each sequence labeled as a unique ID (given as a header).

Input B cell subset file can be prepared by first running the BCR-SORT and labeling each row with the unique sequence ID constituting the lineage.

IgPhyML, a program designed to infer the phylogetic relationships between BCRs, is required to run the command below.

```bash
reroot.py --i <sequence file to construct lineage> --r <B cell subset file> --p <IgPhyML path>
```

