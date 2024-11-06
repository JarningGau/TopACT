import pandas as pd

from scipy import io

from topact.countdata import CountMatrix
from topact.classifier import SVCClassifier, train_from_countmatrix
from topact import spatial
import numpy as np

import matplotlib.pyplot as plt

def readfile(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f]


if __name__ == '__main__':
    # Load single-cell reference

    mtx = io.mmread('scmatrix.mtx').T
    genes = readfile('scgenes.txt')
    labels = readfile('sclabels.txt')

    # Create TopACT object

    sc = CountMatrix(mtx, genes=genes)
    sc.add_metadata("celltype", labels)

    # Train local classifier

    clf = SVCClassifier()
    train_from_countmatrix(clf, sc, "celltype")

    # Load spatial data

    df = pd.read_csv('spatial.csv')

    # Passing in genes will automatically filter out genes that are not in the
    # single cell reference
    sd = spatial.CountGrid.from_coord_table(df, genes=genes, count_col="counts", gene_col="gene")


    # Classify

    sd.classify_parallel(clf, min_scale=3, max_scale=9, num_proc=1, outfile='outfile.npy')


    confidence_mtx = np.load('outfile.npy')

    annotations = spatial.extract_image(confidence_mtx, 0.5)

    np.savetxt("demo-output.txt", annotations, fmt='%1.f')

    plt.imshow(annotations, interpolation='None')
    plt.savefig('demo-output.png')