import geohash
import itertools
import math

from six import operator

from collections import Counter

from lieu.dedupe import Name
from lieu.information_gain import InformationGain
from lieu.spark.geo_word_index import GeoWordIndexSpark


class InformationGainSpark(object):
    @classmethod
    def doc_cooccurrences(cls, docs, has_id=False):
        if not has_id:
            docs = docs.zipWithUniqueId()

        return docs.flatMap(lambda doc_doc_id: [((word, other), 1) for word, other in itertools.permutations(doc_doc_id[0], 2)]) \
                   .reduceByKey(lambda x, y: x + y)

    @classmethod
    def filter_min_doc_count(cls, word_marginals, min_count=2):
        return word_marginals.filter(lambda key_count: key_count[1] >= min_count)

    @classmethod
    def doc_words(cls, docs, has_id=False, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        doc_words = docs.flatMap(lambda doc_doc_id8: [(word, (doc_doc_id8[1], i))
                                                        for i, word in enumerate(doc_doc_id8[0])])
        return doc_words

    @classmethod
    def word_marginal_probs(cls, docs, total_docs, has_id=False, min_count=1):
        if not has_id:
            docs = docs.zipWithUniqueId()

        marginals = docs.flatMap(lambda doc_doc_id9: [(word, 1) for word in set(doc_doc_id9[0])]) \
                        .reduceByKey(lambda x, y: x + y)

        if min_count > 1:
            marginals = cls.filter_min_doc_count(marginals, min_count=min_count)

        return marginals.mapValues(lambda count: float(count) / total_docs)

    @classmethod
    def word_info_gain(cls, doc_cooccurrences, word_marginal_probs, min_count=1):
        num_partitions = doc_cooccurrences.getNumPartitions()

        doc_sum_cooccurrences = doc_cooccurrences.map(lambda word_other_count: (word_other_count[0][0], word_other_count[1])).reduceByKey(lambda x, y: x + y)
        probs = doc_cooccurrences.map(lambda word_other_count10: (word_other_count10[0][0], (word_other_count10[0][1], word_other_count10[1]))).join(doc_sum_cooccurrences).map(lambda word_other_c_xy_c_y: ((word_other_c_xy_c_y[0], word_other_c_xy_c_y[0][0]), float(word_other_c_xy_c_y[0][1]) / word_other_c_xy_c_y[1][1]))

        coo_info_gain = probs.map(lambda word_other_prob: (word_other_prob[0][1], (word_other_prob[0][0], word_other_prob[1]))) \
                             .join(word_marginal_probs) \
                             .map(lambda other_word_p_xy_p_x: (other_word_p_xy_p_x[0][0], math.log(other_word_p_xy_p_x[0][1] / other_word_p_xy_p_x[1][1], 2) * other_word_p_xy_p_x[0][1])) \
                             .reduceByKey(lambda x, y: x + y) \
                             .mapValues(lambda value: max(value, 0.0))
        no_coo_info_gain = word_marginal_probs.subtractByKey(coo_info_gain).mapValues(lambda p_x: math.log(1.0 / p_x, 2))
        word_info_gain = coo_info_gain.union(no_coo_info_gain)

        return word_info_gain.coalesce(num_partitions)

    @classmethod
    def doc_scores(cls, doc_words, word_info_gain):
        num_partitions = doc_words.getNumPartitions()

        doc_word_stats = doc_words.join(word_info_gain).map(lambda word_doc_id_pos_info_gain: (word_doc_id_pos_info_gain[0][0], (word_doc_id_pos_info_gain[0], word_doc_id_pos_info_gain[0][1], word_doc_id_pos_info_gain[1][1])))

        docs_info_gain = doc_word_stats.groupByKey() \
                                       .mapValues(lambda vals: [(word, val) for word, pos, val in sorted(vals, key=operator.itemgetter(1))])

        return docs_info_gain.coalesce(num_partitions)

    @classmethod
    def save(cls, word_info_gain, index_path):
        word_info_gain.map(lambda word_val: '\t'.join(word_val[0], word_val[1])) \
                      .saveAsTextFile(index_path)


class GeoInformationGainSpark(InformationGainSpark, GeoWordIndexSpark):
    @classmethod
    def doc_cooccurrences(cls, docs, geo_aliases=None, has_id=False, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id11: doc_lat_lon_doc_id11[0][1] is not None and doc_lat_lon_doc_id11[0][2] is not None)

        if geo_aliases:
            doc_geohashes = docs.flatMap(lambda doc_lat_lon_doc_id: [(geo, (doc_lat_lon_doc_id[0][0], doc_lat_lon_doc_id[1])) for geo in cls.geohashes(doc_lat_lon_doc_id[0][1], doc_lat_lon_doc_id[0][2])]) \
                                .leftOuterJoin(geo_aliases) \
                                .map(lambda geo_doc_doc_id_geo_alias: (geo_doc_doc_id_geo_alias[1][1] or geo_doc_doc_id_geo_alias[0], (geo_doc_doc_id_geo_alias[0][0], geo_doc_doc_id_geo_alias[0][1])))
        else:
            doc_geohashes = docs.flatMap(lambda doc_lat_lon_doc_id1: [(geo, (doc_lat_lon_doc_id1[0][0], doc_lat_lon_doc_id1[1])) for geo in cls.geohashes(doc_lat_lon_doc_id1[0][1], doc_lat_lon_doc_id1[0][2])])

        return doc_geohashes.flatMap(lambda geo_doc_doc_id: [((geo_doc_doc_id[0], word, other), 1) for word, other in itertools.permutations(geo_doc_doc_id[1][0], 2)]) \
                            .reduceByKey(lambda x, y: x + y)

    @classmethod
    def doc_words(cls, docs, geo_aliases=None, has_id=False, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id12: doc_lat_lon_doc_id12[0][1] is not None and doc_lat_lon_doc_id12[0][2] is not None)

        if geo_aliases:
            doc_geohashes = docs.map(lambda doc_lat_lon_doc_id2: (cls.geohash(doc_lat_lon_doc_id2[0][1], doc_lat_lon_doc_id2[0][2], geohash_precision=geohash_precision), (doc_lat_lon_doc_id2[0][0], doc_lat_lon_doc_id2[1]))) \
                                .leftOuterJoin(geo_aliases) \
                                .map(lambda geo_doc_doc_id_geo_alias3: (geo_doc_doc_id_geo_alias3[1][1] or geo_doc_doc_id_geo_alias3[0], (geo_doc_doc_id_geo_alias3[0][0], geo_doc_doc_id_geo_alias3[0][1])))
        else:
            doc_geohashes = docs.map(lambda doc_lat_lon_doc_id4: (cls.geohash(doc_lat_lon_doc_id4[0][1], doc_lat_lon_doc_id4[0][2], geohash_precision=geohash_precision), (doc_lat_lon_doc_id4[0][0], doc_lat_lon_doc_id4[1])))

        doc_words = doc_geohashes.flatMap(lambda geo_doc_doc_id13: [((geo_doc_doc_id13[0], word), (geo_doc_doc_id13[1][1], pos))
                                                                        for pos, word in enumerate(geo_doc_doc_id13[1][0])])
        return doc_words

    @classmethod
    def word_marginal_probs(cls, docs, total_docs_by_geo, geo_aliases=None, has_id=False, min_count=1, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id14: doc_lat_lon_doc_id14[0][1] is not None and doc_lat_lon_doc_id14[0][2] is not None)
        num_partitions = docs.getNumPartitions()

        if geo_aliases:
            doc_geohashes = docs.flatMap(lambda doc_lat_lon_doc_id5: [(gh, (doc_lat_lon_doc_id5[0][0], doc_lat_lon_doc_id5[1])) for gh in cls.geohashes(doc_lat_lon_doc_id5[0][1], doc_lat_lon_doc_id5[0][2])]) \
                                .leftOuterJoin(geo_aliases) \
                                .map(lambda geo_doc_doc_id_geo_alias6: (geo_doc_doc_id_geo_alias6[1][1] or geo_doc_doc_id_geo_alias6[0], (geo_doc_doc_id_geo_alias6[0][0], geo_doc_doc_id_geo_alias6[0][1])))
        else:
            doc_geohashes = docs.flatMap(lambda doc_lat_lon_doc_id7: [(gh, (doc_lat_lon_doc_id7[0][0], doc_lat_lon_doc_id7[1])) for gh in cls.geohashes(doc_lat_lon_doc_id7[0][1], doc_lat_lon_doc_id7[0][2])])

        marginals = doc_geohashes.flatMap(lambda geo_doc_doc_id15: [((geo_doc_doc_id15[0], word), 1) for word in set(geo_doc_doc_id15[1][0])]) \
                                 .reduceByKey(lambda x, y: x + y)

        if min_count > 1:
            marginals = cls.filter_min_doc_count(marginals, min_count=min_count)

        return marginals.map(lambda geo_word_count: (geo_word_count[0][0], (geo_word_count[0][1], geo_word_count[1]))) \
                        .join(total_docs_by_geo) \
                        .map(lambda geo_word_count_num_docs: ((geo_word_count_num_docs[0], geo_word_count_num_docs[0][0]), float(geo_word_count_num_docs[0][1]) / geo_word_count_num_docs[1][1])) \
                        .coalesce(num_partitions)

    @classmethod
    def word_info_gain(cls, doc_cooccurrences, word_marginal_probs):
        num_partitions = word_marginal_probs.getNumPartitions()

        doc_sum_cooccurrences = doc_cooccurrences.map(lambda geo_word_other_count: ((geo_word_other_count[0][0], geo_word_other_count[0][1]), geo_word_other_count[1])).reduceByKey(lambda x, y: x + y)
        probs = doc_cooccurrences.map(lambda geo_word_other_count16: ((geo_word_other_count16[0][0], geo_word_other_count16[0][1]), (geo_word_other_count16[0][2], geo_word_other_count16[1]))).join(doc_sum_cooccurrences).map(lambda geo_word_other_c_xy_c_y: ((geo_word_other_c_xy_c_y[0][0], geo_word_other_c_xy_c_y[0][1], geo_word_other_c_xy_c_y[0][0]), float(geo_word_other_c_xy_c_y[0][1]) / geo_word_other_c_xy_c_y[1][1]))

        coo_info_gain = probs.map(lambda geo_word_other_prob: ((geo_word_other_prob[0][0], geo_word_other_prob[0][2]), (geo_word_other_prob[0][1], geo_word_other_prob[1]))) \
                             .join(word_marginal_probs) \
                             .map(lambda geo_other_word_p_xy_p_x: ((geo_other_word_p_xy_p_x[0][0], geo_other_word_p_xy_p_x[0][0]), math.log(geo_other_word_p_xy_p_x[0][1] / geo_other_word_p_xy_p_x[1][1], 2) * geo_other_word_p_xy_p_x[0][1])) \
                             .reduceByKey(lambda x, y: x + y) \
                             .mapValues(lambda value: max(value, 0.0))
        no_coo_info_gain = word_marginal_probs.subtractByKey(coo_info_gain).mapValues(lambda p_x: math.log(1.0 / p_x, 2))
        word_info_gain = coo_info_gain.union(no_coo_info_gain)

        return word_info_gain.coalesce(num_partitions)

    @classmethod
    def doc_scores(cls, doc_words, word_info_gain):
        num_partitions = doc_words.getNumPartitions()

        doc_word_stats = doc_words.join(word_info_gain).map(lambda geo_word_doc_id_pos_info_gain: (geo_word_doc_id_pos_info_gain[0][0], (geo_word_doc_id_pos_info_gain[0][1], geo_word_doc_id_pos_info_gain[0][1], geo_word_doc_id_pos_info_gain[1][1])))

        docs_info_gain = doc_word_stats.groupByKey() \
                                       .mapValues(lambda vals: [(word, info_gain) for word, pos, info_gain in sorted(vals, key=operator.itemgetter(1))])

        return docs_info_gain.coalesce(num_partitions)

    @classmethod
    def save(cls, word_info_gain, index_path):
        word_info_gain.map(lambda geo_word_val: '\t'.join([geo_word_val[0][0], geo_word_val[0][1], geo_word_val[1]])) \
                      .saveAsTextFile(index_path)
