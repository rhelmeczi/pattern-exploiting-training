# Pattern Exploiting Training

This repository is based largely on the following references:

* Schick, T., and Schütze, H. (2020a). *Exploiting cloze questions for few shot text classification and natural language inference*.
arXiv preprint arXiv:2001.07676.
* Schick, T., and Schütze, H. (2020b). *It's not just size that matters: Small language models are also few-shot learners*. arXiv preprint arXiv:2009.07118.

## Usage

The implementation is designed to run on Google colab. It requires usage of
Google drive.

1. On Google Drive under `My Drive`, create a folder named `PET`.
2. Upload `src/` and `pattern-exploiting-training.ipynb` from this repository
to the newly created `PET` folder on your drive.

The code also assumes access to GPU hardware acceleration. To activate
this in Google colab, open `pattern-exploiting-training.ipynb`,
go to Edit > Notebook Settings and set Hardware accelerator from None to
GPU.

You are now fully setup to run the implementation.

## Notes

* You should be able to run the entire notebook in around 20 minutes.
* Progress bars are disabled as much as possible in the notebook to keep
it clear. Feel free to enable them as you execute the notebook.
* I've added numerous functions to `srcs/helper_functions` to clean up
the notebook. Each of these functions is well documented and you're welcome
to experiment as you see fit. Keep in mind that they are largely targeted
towards solving this project, meaning they might not be as flexible to
solving different problems.
* The dataset is loaded from `datasets` automatically. If you have trouble
accessing the Yelp polarity dataset, it is widely available online.
* There are several bugfixes and warnings that I've resolved throughout
the PET code. I've also hidden some progress bars that were needlessly
cluttering the document. As a result, if you'd like to truly experiment
with PET, I recommend checking out the repository referenced in both
Schick and Schütze papers.
