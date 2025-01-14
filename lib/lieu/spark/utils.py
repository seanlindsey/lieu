

class IDPairRDD(object):
    @classmethod
    def join_pairs(cls, pairs, kvs):
        result = pairs.join(kvs) \
                      .map(lambda k1_k2_v1: (k1_k2_v1[1][0], (k1_k2_v1[0], k1_k2_v1[1][1])))

        num_partitions = result.getNumPartitions()

        return result.join(kvs) \
                     .map(lambda k2_k1_v1_v2: ((k2_k1_v1_v2[0][0], k2_k1_v1_v2[0]), (k2_k1_v1_v2[0][1], k2_k1_v1_v2[1][1]))) \
                     .coalesce(num_partitions)

    @classmethod
    def join_pairs_multi_kv(cls, pairs, kvs1, kvs2):
        result = pairs.join(kvs1) \
                      .map(lambda k1_k2_v11: (k1_k2_v11[1][0], (k1_k2_v11[0], k1_k2_v11[1][1])))

        num_partitions = result.getNumPartitions()

        return result.join(kvs2) \
                     .map(lambda k2_k1_v1_v22: ((k2_k1_v1_v22[0][0], k2_k1_v1_v22[0]), (k2_k1_v1_v22[0][1], k2_k1_v1_v22[1][1]))) \
                     .coalesce(num_partitions)
