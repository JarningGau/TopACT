# TopACT: Topological Automatic Cell Types

## Note

This repository is forked from https://gitlab.com/kfbenjamin/topact. 



---

TopACT is a method for annotating subcellular spatial transcriptomics data with cell types inferred from a single-cell reference data set. See the paper "Multiscale topology classifies and quantifies cell types in subcellular spatial transcriptomics" for details.

This repository contains a Python implemetation of TopACT designed to run on square spatial grids.

## Installation

Install directly from the repository using pip:

```pip install git+https://gitlab.com/kfbenjamin/topact.git```

Installation should take no more than a few minutes on a standard desktop computer.

For development purposes, the package is maintained with Poetry. So you can
alternatively install Poetry and then run `poetry install` in the cloned repository.


### MacOS

On systems running Apple silicon (i.e. any Mac system released since 2020), poetry may fail to install the scipy dependency, citing a missing BLAS implementation.

This can be fixed with [the following workaround](https://github.com/numpy/numpy/issues/17784#issuecomment-729950525):

```bash
brew install openblas
OPENBLAS="$(brew --prefix openblas)" poetry install
```
## Dependencies

TopACT requires Python 3.10.

The following dependencies are also required and will be automatically installed alongside TopACT

- scipy >=1.8
- numpy >=1.21.4
- pandas >=1.3.4
- scikit-learn >=1.0.1
- py-find-1st >=1.1.5
- scikit-image >=0.19.2

## Example usage

The `demo` directory contains a small example data set and code to help you get TopACT up and running. It should only take a few seconds to run on a standard laptop computer, and example output files are provided for verification.

Below we explain basic usage of the package.

### Single-cell data

The following code creates a support-vector classifier from a single-cell data set. It assumes the following data are in the name space:

- `mtx`: a sparse Scipy matrix of shape (`num_samples, num_genes`)
- `genes`: a list of gene identifiers to be interpreted as column labels
- `labels`: a list of cell type annotations to be interpreted as row labels

```python
from topact.countdata import CountMatrix
from topact.classifier import SVCClassifier, train_from_countmatrix

# Create topact CountMatrix
sc = CountMatrix(mtx, genes=genes)

# Annotate single-cell data with labels under the heading "celltype"
sc.add_metadata("celltype", labels)

# Create new classifier
clf = SVCClassifier()

# Train classifier from single-cell data and annotations
train_from_countmatrix(clf, sc, "celltype")
```

### Spatial data

Now, we need to provide some spatial data to be classified. Suppose `df` is a Pandas `DataFrame` with the following columns:

- `"x"` corresponding to spatial x-coordinate;
- `"y"` corresponding to spatial y-coordinate;
- `"gene"` corresponding to gene ID (as in `scdata` above);
- `"counts"` corresponding to number of transcript counts.

In particular, each row of `df` records the instance of a single gene at a single spatial location. We can classify the cell types using `clf` using the following code.

```python
import numpy as np
from topact import spatial

# Convert to CountGrid object, filtering out genes not in the earlier list
sd = spatial.CountGrid.from_coord_table(df, 
                                        genes=genes,
                                        count_col="counts",
                                        gene_col="gene"
                                        )

# Produce multiscale confidence matrix and output to outfile.npy
sd.classify_parallel(clf,
                     min_scale=3, # Min nbhd radius
                     max_scale=9, # Max nbhd radius
                     num_proc=8, # Number of workers
                     outfile='outfile.npy', # Output file
                     verbose=False # Print progress
                     )

# Extract classifications from confidence matrix
confidence_mtx = np.load('outfile.npy')
threshold=0.5
annotations = spatial.extract_image(confidence_mtx, threshold)
```

The output `annotations` is a Numpy array corresponding to the underlying shape of the spatial data. Its entries are integers corresponding to the cell type labels used to train the single-cell data. The list of these classes is stored in `clf.classes`. So, for example, if `annotations[i,j] == x` then the cell type assigned to the (i,j)-th coordinate is `clf.classes[x]`. If no cell type is assigned then the value is `np.nan`.

Note that the grid is translated so that the bottom-left most corner of the original spatial data is at (0,0).

## Documentation

Further documentation can be found on [Read the Docs](https://topact.readthedocs.io/en/latest/).
