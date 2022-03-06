import numpy as np
import scipy.sparse as sp
from numpy.random import rand
import time


def normalize(A):
    #Find the row scalars as a Matrix_(n,1)
    row_sum = sp.csr_matrix(A.sum(axis=1))
    row_sum.data = 1/row_sum.data

    #Find the diagonal matrix to scale the rows
    row_sum = row_sum.transpose()
    scaling_matrix = sp.diags(row_sum.toarray()[0])

    return scaling_matrix.dot(A)
# https://stackoverflow.com/questions/12305021/efficient-way-to-normalize-a-scipy-sparse-matrix
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.sum.html

def transmatrix():
    row = []
    col = []
    value = []

    transdatapath = "data/transition.txt"

    with open(transdatapath, "r") as f:
        rough_transdata = f.read().split("\n")
        # split function adds '' at the end of list
        rough_transdata = rough_transdata[:-1]

    for line in rough_transdata:
        temp = line.split(" ")
        # python index num begins from 0
        row.append(int(temp[0]) - 1)
        col.append(int(temp[1]) - 1)
        value.append(int(temp[2]))

    dimnum = max(max(row), max(col)) + 1

    # construct a transition matrix
    transmat = sp.csr_matrix((value, (row, col)), shape=(dimnum, dimnum))
    transmat = normalize(transmat)

    # through dot prod, we can find indices satisfying all element is 0
    rowinfo = transmat.dot(np.ones(dimnum))
    row_0 = abs(np.trunc(np.ones(dimnum, dtype=int) - rowinfo).astype(int))

    transmat = np.transpose(transmat)

    return transmat,dimnum,row_0

def g_pagerank():
    iter_num = 10000000
    error = 1e-15
    alpha = 0.8

    transmat, dimnum,row_0 = transmatrix()
    initvec = np.random.rand(dimnum)
    initvec /= initvec.sum()

    p_0 = np.ones(dimnum)/dimnum
    # transmat = np.asarray(transmat).transpose()

    start_pagerankT = time.time()
    for i in range(1, iter_num + 1):
        row_0_data = sum(np.multiply(row_0, initvec))
        temp = alpha * transmat * initvec + alpha * row_0_data * p_0 + (1 - alpha) * p_0
        if pow(temp-initvec, 2).sum(axis=0) ** 0.5 < error:
            break
        initvec = temp
    pagerank = temp

    end_pagerankT = time.time()
    pagerankT = end_pagerankT - start_pagerankT

    print("pagerank takes %s seconds" % pagerankT)

    return pagerank


def record():
    with open("GPR.txt", "w") as f:
        document_ID = 0
        pagerank = g_pagerank()
        for i in pagerank:
            document_ID += 1
            f.write(str(document_ID) + " " + str(i) + "\n")


if __name__ == "__main__":
    f = record()
    print("completed\n")
