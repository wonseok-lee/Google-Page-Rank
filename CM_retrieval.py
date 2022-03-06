import numpy as np
from transition import *
import time
import indri
import TSPR



def GPRCM():
    alpha = 0.9
    startNS = time.time()
    pagerank = g_pagerank()
    with open('GPR_CM.txt', 'w') as f:

        for i in dict:
            queryID = dict[i]["queryID"]
            doc_id = dict[i]["docid"]

            gpr_value = pagerank[doc_id]
            gpr_value = gpr_value * 1e+5
            gpr_value = np.exp(gpr_value)
            gpr_norm = [(float(j) - np.min(gpr_value) + 0.01) / (np.max(gpr_value) - np.min(gpr_value)) for j in
                        gpr_value]

            indri_score = dict[i]["content"]
            indri_norm = [(float(j) - np.min(indri_score) + 0.01) / (np.max(indri_score) - np.min(indri_score)) for j in
                          indri_score]
            indri_norm = np.exp(indri_norm) * np.abs(np.multiply(indri_norm, 1e+5))
            indri_norm = indri_norm + np.asarray(indri_score)

            score_1 = np.asarray(gpr_norm)
            score_2 = np.asarray(indri_norm)
            score_sum = alpha * score_1 + (1 - alpha) * score_2

            score = np.argsort(score_sum)[::-1].tolist()

            doc_id_array = np.asarray(doc_id)
            pagerank_num = doc_id_array[score]

            n = 1

            for k in pagerank_num:
                f.write("{} Q0 {} {} {} run-1\n".format(queryID, k + 1, n, score_sum[doc_id.index(k)]))
                n = n + 1
    endNS = time.time()
    time_NS = endNS - startNS
    print("GPR_CM takes %s seconds" % time_NS)
    print("GPR_CM completed \n")


def PTSPRWS():
    dict = indri.indri_info()
    ptspr = TSPR.ptspr_on()
    alpha = 0.5
    startWS = time.time()
    with open('PTSPR_CM.txt', 'w') as f:
        query = -1
        for i in dict:
            query = query + 1
            query_id = dict[i]["queryID"]
            doc_id = dict[i]["docid"]

            ptspr_value = ptspr[query][doc_id]
            ptspr_value = ptspr_value * 1e+5
            ptspr_value = np.exp(ptspr_value)
            ptspr_norm = [(float(j) - np.min(ptspr_value) + 0.01) / (np.max(ptspr_value) - np.min(ptspr_value)) for j in
                          ptspr_value]

            indri_score = dict[i]["content"]
            indri_norm = [(float(j) - np.min(indri_score) + 0.01) / (np.max(indri_score) - np.min(indri_score)) for j in
                          indri_score]
            indri_norm = np.exp(indri_norm) * np.abs(np.multiply(indri_norm, 1e+5))
            indri_norm = indri_norm + np.asarray(indri_score)

            score_1 = np.asarray(ptspr_norm)
            score_2 = np.asarray(indri_norm)
            score_sum = alpha * score_1 + (1 - alpha) * score_2
            score = np.argsort(score_sum)[::-1].tolist()

            doc_id_array = np.asarray(doc_id)
            ptspr_num = doc_id_array[score]
            # print(len(pagerank_num))
            n = 1

            for k in ptspr_num:
                f.write("{} Q0 {} {} {} run-1\n".format(query_id, k + 1, n, score_sum[doc_id.index(k)]))
                n = n + 1

    endWS = time.time()
    time_WS = endWS - startWS
    print("PTSPR_CM takes %s seconds" % time_WS)
    print("PTSPR_CM completed \n")

def QTSPRCM():
    dict = indri.indri_info()
    qtspr = TSPR.qtspr_on()
    alpha = 0.5
    startWS = time.time()
    with open('QTSPR_CM.txt', 'w') as f:
        query = -1
        for i in dict:
            query = query + 1
            query_id = dict[i]["queryID"]
            doc_id = dict[i]["docid"]

            qtspr_value = qtspr[query][doc_id]
            qtspr_value = qtspr_value * 1e+5
            qtspr_value = np.exp(qtspr_value)
            qtspr_norm = [(float(j) - np.min(qtspr_value) + 0.01) / (np.max(qtspr_value) - np.min(qtspr_value)) for j in
                          qtspr_value]

            indri_score = dict[i]["content"]
            indri_norm = [(float(j) - np.min(indri_score) + 0.01) / (np.max(indri_score) - np.min(indri_score)) for j in
                          indri_score]
            indri_norm = np.exp(indri_norm) * np.abs(np.multiply(indri_norm, 1e+5))
            indri_norm = indri_norm + np.asarray(indri_score)

            score_1 = np.asarray(qtspr_norm)
            score_2 = np.asarray(indri_norm)
            score_sum = alpha * score_1 + (1 - alpha) * score_2
            score = np.argsort(score_sum)[::-1].tolist()

            doc_id_array = np.asarray(doc_id)
            ptspr_num = doc_id_array[score]
            # print(len(pagerank_num))
            n = 1

            for k in ptspr_num:
                f.write("{} Q0 {} {} {} run-1\n".format(query_id, k + 1, n, score_sum[doc_id.index(k)]))
                n = n + 1

    endWS = time.time()
    time_WS = endWS - startWS
    print("QTSPR_CM takes %s seconds" % time_WS)
    print("QTSPR_CM completed \n")
