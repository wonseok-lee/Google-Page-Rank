import numpy as np
from transition import *
import scipy.sparse as sp
import time

def normalize(A):
    row_sum = sp.csr_matrix(A.sum(axis=1))
    row_sum.data = 1/row_sum.data

    row_sum = row_sum.transpose()
    scaling_matrix = sp.diags(row_sum.toarray()[0])

    return scaling_matrix.dot(A)

def qtspr_trans():
    topic_row = []
    topic_col = []
    topic_value = []
    topicIDpath = "data/doc_topics.txt"

    with open(topicIDpath, "r") as f:
        rough_topicID = f.read().split("\n")
        rough_topicID = rough_topicID[:-1]

    for line in rough_topicID:
        temp = line.split(" ")
        topic_row.append(int(temp[1])-1)
        topic_col.append(int(temp[0])-1)
        topic_value.append(1)

    row_num = max(topic_row)+1
    col_num = max(topic_col)+1

    # construct a transition matrix
    topic_transmat = sp.csr_matrix((topic_value, (topic_row, topic_col)), shape=(row_num, col_num)) # 12x81433 matrix
    topic_transmat = normalize(topic_transmat)

    return topic_transmat,col_num

def qtspr_off():
    alpha = 0.8
    beta = 0.1
    gamma = 0.1
    iter_num = 2
    error = 1e-15
    topic_pagerank = []

    transmat,dimnum,row_0 = transmatrix()
    topic_transmat, col_num = qtspr_trans()
    p_0 = np.ones(dimnum) / dimnum
    topic_num = topic_transmat.shape[0]

    start_qtspr = time.time()
    for i in range(0,topic_num):
        p_t = topic_transmat[i]
        initvec = np.random.rand(dimnum)
        initvec /= initvec.sum()
        for j in range(1, iter_num+1):
            row_0_data = sum(np.multiply(row_0, initvec))
            temp_new = alpha * transmat * initvec + alpha * row_0_data * p_0 + beta * p_t + gamma * p_0
            temp_new = np.asarray(temp_new).squeeze()
            initvec = np.asarray(initvec)
            if pow(temp_new-initvec, 2).sum(axis=0) ** 0.5 < error:
                break
            initvec = temp_new
        topic_pagerank.append(initvec)
    end_qtspr = time.time()

    qtsprT = end_qtspr - start_qtspr

    print("QTSPR takes %s seconds" % qtsprT)

    return topic_pagerank


def qtspr_on():
    usertopicpath = "data/query-topic-distro.txt"

    with open(usertopicpath, "r") as fp:
        rough_usertopic = fp.read().split("\n")
        rough_usertopic = rough_usertopic[:-1]

    userIDp_row = []
    queryp = []
    userIDp_prob = []

    for line1 in rough_usertopic:
        temp1 = line1.split(" ")
        userIDp_row.append(temp1[0])
        queryp.append(temp1[1])
        temp1 = temp1[2:]
        # temp2 = []
        temp3 = []
        for line2 in temp1:
            temp2 = line2.split(":")
            temp3.append(temp2[1])
        userIDp_prob.append(temp3)

    userIDp_prob = np.asfarray(userIDp_prob)
    topic_pagerank = qtspr_off()

    qtspr = userIDp_prob.dot(topic_pagerank)

    return qtspr


def record():
    with open("QTSPR-U2Q1.txt", "w") as f:
        document_ID = 0
        qtspr = qtspr_on()
        for i in qtspr:
            document_ID += 1
            f.write(str(document_ID) + " " + str(i) + "\n")


