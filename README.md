This repository accompanies my MSc thesis 'Probabilistic Head Pose Estimation'. It contains the main Python scripts needed to reproduce results, as well as a copy of the thesis for reference.

The data getter/generator is designed for the AFLW dataset, which is available for research purposes from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/.
Annotations are extracted from the SQLite database. To use with another dataset, the data getter will need to be adapted.

To use with AFLW, use the script that comes with the unzipped dataset to extract scaled images to:
[project directory]\AFLW\data\rescaled\[image resolution]\all

e.g. 32x32 images on my Windows 10 computer are in C:\Users\Ray\py3\Headpose\AFLW\data\rescaled\32\all.

The AFLW directory in this repository is empty. If you wish to save your results, add a 'saves/mainresults' folder in the root directory.

With the images and database correctly located, the main training file train.py will begin training iteratively over all network architectures and resolutions specified.
Augmentation.py and numcomponents.py run other training procedures.
Display_all_results.py does just that.
Finaltest.py loads a trained model (currently the Inception-v3, input resolution 149x149 pixels) and evaluates it on the unused final test set.
