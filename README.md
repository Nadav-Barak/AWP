# AWP
This project contains the implantation of the AWP algorithm and the code used to prefomed the experiments as described in the paper "Approximating a Target Distribution using Weight Queries" by Nadav Barak and Sivan Sabato.

1. Files and folders in archive:
- README - this file
- Code - A folder with the code for running all the experiments
and the AWP algorithm.
- ExperimentsResults - A folder with the full results for each experiment
reported in the paper; see details below.
- InputFiles - A folder with preprocessed files; see details below. 




2. Dependencies:
The code is run using python 3.7. 
Required python packages: numpy, pandas, opencv, anytree, scipy. 
All command lines below assume that the working directory is 'Code'.

3. Common command line arguments:
The following command line arguments are used by several commands below.
Specific arguments are detailed under each command.

- weight_file and tree_file - These arguments specify files describing the
  true weights of data set examples and the hierarchical tree defined over the
  data set. For pre-calculated input files, see "6. Pre-calculated input
  files". To calculate the input files from scratch, see "7. Generating the
  input files from a data set".

- output_file - the csv file in which the output of the run will be recorded. 





4. Running AWP:
The command line for running AWP is:

python3.7 AWP.py weight_file tree_file K delta beta 

K , delta and beta are the input arguments of AWP as defined in Alg. 1 in the paper.

- K - pruning size (an integer larger than 1)
- delta - confidence parameter in (0,1), 
- beta - trade-off parameter, must be larger than 1.






5. Reproducing the reported experiments:
The command line for running an experiment is:

python3.7 Experiment.py weights_file tree_file output_file stepsize largest_K repetitions

This command tests the algorithms AWP, WEIGHT, UNIFORM and EMPIRICAL. It
creates an output csv file containing the results of the experiment. In the
output file, each row gives the results of a single algorithm, for a single
repetition, for the predefined pruning sizes.

- stepsize, largest_K - Integers which specify which pruning sizes will be tested in the experiment. The tested sizes will be:
stepsize, 2stepsize, ... ,n*stepsize
where n is the largest integer such that n*stepsize <= largest_K.

- repetitions - Integer, number of times to repeat the experiment.





6. Pre-calculated input files and experiment results
InputFiles/Trees includes csv files describing the hierarchical trees
constructed using the Ward method for MNIST and Caltech256,
which can be used for the input parameter tree_file.

InputFiles/Weights includes files describing the example weights
for the reported experiments.

ExperimentsResults includes the output files of the reported experiments.

The naming conventions for the files in InputFiles/Weights and in
ExperimentsResults are as follows, where 'D' stands for a data set (Caltech or
MNIST), and 'X' and 'Y' stand for parameters:

- D_brightness_NX.csv - The experiment with weights allocated by brightness
for the 'D' dataset with N=X.

- D_randomclasses_NX_vY.csv - The experiment with weights allocated by class
for the 'D' data set with N=X, using the Y'th random order of bins
(and class allocation to bins for Caltech) out of the three that were tested.
The random settings that were used are available at
Code/Randomclass_weight_orders.txt

- Caltech_DA_office.csv - The domain adaptation experiment where the target
weights where determined by the Office dataset

- Caltech_DA_bing_full.csv - The domain adaptation experiment where the target
weights where determined by the full Bing dataset

- Caltech_DA_bing_X.csv - The domain adaptation experiment where the target
weights where determined by the 'X' super category from Bing







7. Generating the input files from a data set:

(i) Download the input data set (e.g., Caltech256, MNIST) from the given url.
If the data set is given as a folder with images, convert it to a single csv
file using the following command:

python3.7 images_to_vector.py images_dir output_file

- images_dir - the directory containing the data set images.
The images must reside in sub-folders according to their allocation to classes.



(ii) Generate the hierarchical tree: 

python3.7 Create_hierarchical_tree.py data_file linkage_method output_file. 

- data_file - the csv file containing the vectorized images (the output of images_to_vector.py)

- linkage_method -  the type of linkage method to use for creating the hierarchical tree. The valid inputs are those allowed by the scipy linkage function. To reproduce the experiments reported in the paper, set linkage_method to ward.



(iii) Generate a csv file which maps each input example to its weight, using one of the following options, based on the required experiment. 
In the commands below, the following additional arguments are used:

- data_file - same as in (ii) above
- N - A positive number. The value of N (the weight multiplication factor) to set in the experiment. 
- images_dir - same as in (i) above.

The options for generating weights are:

(a) Setting the weights based on bins defined by image brightness:

python3.7 Create_brightness_weights.py data_file N output_file

(b) For MNIST, setting the weights based on bins defined by example class:

python3.7 Create_class_weights_mnist.py data_file N output_file

This command allocates the weights for MNIST using bins defined by the data set
classes, in a random order. The random orders used in the experiments reported
in the paper are listed in Code\Randomclass_orders.txt

(c) For Caltech256, setting the weights based on bins defined by example classes:
python3.7 Create_class_weights_caltech.py images_dir N output_file

This command allocates the weights for Caltech256 using bins defined by the
data set classes, randomly allocated into 10 bins. The random bins used in the
experiments reported in the paper are listed in Code\Randomclass_orders.txt

(d) For the domain adaptation experiments: 
First, download the target data set. Then, run the command:

python3.7 Create_1NN_weights data_file target_images_dir output_file [classes_to_include]

- target_images_dir - the directory which includes the images of the target data
set, arranged in the same directory structure as defined for images_dir above.

- classes_to_include - an optional argument stating the names of classes we wish
to use from the target data. If not specified, all images in target_images_dir
are used.  Each class is assumed to reside in a separate folder under
target_images_dir, and the name of the folder is considered the name of the class. Each class is a separate input argument, with a space between class names. 





8. Example
The following series of commands runs the Caltech256 brightness experiment with N=4, assuming the folders containing the images for each class are in the folder 'Caltech' and that the working directory is 'Code'.

python3.7 images_to_vector.py Caltech/256_ObjectCategories/ Caltech.csv
python3.7 Create_hierarchical_tree.py Caltech.csv ward   ../InputFiles/Trees/Caltech_ward_links.csv
python3.7 Create_brightness_weights.py Caltech.csv 4 ../InputFiles/Weights/Caltech_brightness_N4.csv
python3.7 Experiment.py ../InputFiles/Weights/Caltech_brightness_N4.csv ../InputFiles/Trees/Caltech_ward_links.csv ../CaltechBrightnessN4.csv 3 60 10
