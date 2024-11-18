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

One of the challenges of such a ML task is that we are have only partial
knowledge of the toxins.

## Guidelines

Projects can be done either by groups of 2 students or alone. They will be
graded based on (1) a report; (2) the first interview; (3) possible, if needed
for clarifications, a second interview.

I will be available to answer questions about the data and the project during
office hours on zoom friday afternoons, from 3.30pm to 5pm. This office hours
need to be booked at least a day before. It will be on a first come first
serve basis. The first interview will happen during one of these slots.

The final report is due **Friday, December 20th**, and should be send by email
at nelle.varoquaux@univ-grenoble-alpes.fr.  Instructions on the final report
will be given during the second phase of the project.

## The dataset

We downloaded the whole set of complete prokaryotic genomes in 2021 (30,000
genomes in total). We then annotated each protein of each genome as being 
a known effector or not. Specifically, we have a large database of proteins
validated as being effectors, and we look through sequence similarity hits in
all the proteins of all the organism we have. 

The goal of the project is to predict whether a protein an effector or not,
and specifically, we wish to discover novel effectors family. An effector
family can be defined as all the proteins that are a sequence similarity match
to a known proteins on either the whole protein length, or a "domain" of the
protein, e.g., a substring of the amino acid chain.

The data handed in to you is:

    - `training_pos_features.csv` contains the features extracted for a subset
      of the proteins that are known toxins.
    - `training_pos_features_sequences.csv` contains the protein sequences for
      subset described above.
    - `training_pos_features_sequences.fasta` also contains the protein
      sequences for the subset described above, but in fasta format (which is
      the default format for DNA and protein sequences in biology.
    - `training_pos_labels.csv` contains the subset of labels corresponding to
      the selected proteins files.

    - and the equivalent files for proteins that were not successfully
      annotated as toxins (except for labels, as this file would be entirely
      empty).

The features contains a series of features extracted from the protein sequence
(amino acid proportion, di-amino acid proportion, entropy, bio physical
proporties, shallow learning embedding "word 2 vec" approach, deep learning
ESM approach, etc). See below for more information on the features.

## Preliminary analysis of the dataset and analysis plan

The first part of this project is to perform a preliminary analysis of the
dataset. You will do this by exploring the `training_pos_labels.csv` of the
training data as well as performing statistics on the features
`training_pos_features.csv` and `training_others_features.csv`

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

 
## Annexe

### Feature extraction

From each protein's amino acid sequence, we have extracted features using (1)
specific knowledge of the field of protein function annotation; (2) embeddings
of the protein's amino acid sequence using both shallow networks and LLMs.

The columns of the feature data matrix are named through a convention that
allows to map to a type of feature:

- Columns starting with "G1" are amino-acid proportions, the equivalent of
  bags of words for protein sequences.
- Columns starting with "G2" are di-amino acid proportions;
- Columns starting with "G3" are physico-chemical properties extracted from
  the protein sequences: polarity, hydrophobicity, etc
- Columns starting with "G4" are C-triads, a dimensionality reduction of the
  tri-amino-acid decomposition.
- Columns starting with "G5" are shallow embeddings "word2vec"
- Columns starting with "G6" corresponds to the shannon entropy on bits and
  pieces of the protein sequences: we split the protein sequences in chunks,
  and compute the shannon entropy.
- Columns starting with "G7" are global protein parameters inferred based on
  amino acid compositions: molecular weight,  isoelectric point, molar
  extinction coefficient , etc
- Columns startinig with "G8" are quasi order features 
- Columns startinig with "G10" 
