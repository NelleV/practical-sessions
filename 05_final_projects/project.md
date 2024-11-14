# Predicting protein function from its sequence

Predicting the function of a protein is one core challenge in biology. Protein
function is typically inferred using sequence comparison with proteins whose
function is known. This is problematic in many ways. In particular, proteins
can have very different sequences, and yet share the same function. Machine
learning approaches are thus more and more investigated to go beyond sequence
similarity when annotating protein functions in new organism.

This project is about investigating machine learning strategies to infer
whether a protein is an effector or not. An effector is a protein that is
secreted by the bacteria. They consist mostly of toxins. Hence, discovering
novel effectors is of great interest, both to better understand how bacteria
interact in their environment, but also for health and ecological purposes.

## The dataset

We downloaded the whole set of complete prokaryotic genomes in 2021 (30,000
genomes in total). We then annotated each protein of each genome as being 
a known effector or not. Specifically, we have a large database of proteins
validated as being effectors, and we look through sequence similarity hits in
all the proteins of all the organism we have. 

The goal of the project is to predict whether a protein an effector or not,
and specifically, we wish to discover novel effectors family. An effector
family can be defined as all the proteins that are a sequence similarity match
to a known proteins.

The data handed in to you is:

    - `effectors.csv`: this is a dataframe where each row corresponds to a
      protein (e.g., `GCA_000189435.3_ASM18943v3_1443` and each column to a
      known effector/toxin. A hit to a known toxin is annotated as True (or
      1).     
    - `training_pos_features.csv` contains the features extracted for a subset
      of the proteins that are known toxins.
    - `training_pos_features_sequences.csv` contains the protein sequences for
      subset described above.
    - `training_pos_features_sequences.fasta` also contains the protein
      sequences for the subset described above, but in fasta format (which is
      the default format for DNA and protein sequences in biology.
    - `training_pos_labels.csv` contains the subset of labels corresponding to
      the selected proteins files.

    - and the equivalent files for negative samples (except for labels, as
      these are all the hits that did not have a match to our known protein
      database.

The features contains a series of features extracted from the protein sequence
(amino acid proportion, di-amino acid proportion, entropy, bio physical
proporties, shallow learning embedding "word 2 vec" approach, deep learning
ESM approach, etc).


The data can be downloaded
[here](https://filesender.renater.fr/?s=download&token=b28c74b7-7bad-492a-8c9c-5d0119fd93b0)

## Preliminary analysis of the dataset and analysis plan

The first part of this project is to perform a preliminary analysis of the
dataset. You will do this by exploring the `training_pos_labels.csv` of the
training data as well as performing statistics on the features
`training_pos_features.csv` and `training_neg_features.csv`

- How many proteins are in this dataset?
- What is the proportion of positive labels in the dataset? What is the
  proportion of negative labels? What metric(s) would be appropriate to
  validate the machine learning pipeline? Justify each of the metrics used.
- Now, looking at the complete list of labels, what is the number of elements
  in each category? What is the proportion of proteins labeled in two
  categories?
- Identify and describe three challenges when working on this machine learning
  problem.
- Devise (on paper) a cross-validation strategy that would enable to check
  that the machine learning pipeline would extend to effectors **not**
  annotated in our dataset. You can write a pseudo-algorithm describing the
  cross validation strategy, or draw a schema explain your though. Explain how
  this cross validation strategy is a good way to check for generalization in
  this setting.

