import geohash
import math
import six

from six import operator

from collections import Counter

from lieu.tfidf import TFIDF
from lieu.dedupe import Name
from lieu.spark.geo_word_index import GeoWordIndexSpark


class TFIDFSpark(object):
    @classmethod
    def doc_words(cls, docs, has_id=False):
        if not has_id:
            docs = docs.zipWithUniqueId()

        doc_words = docs.flatMap(lambda doc_doc_id: [(word, (doc_doc_id[1], pos))
                                                        for pos, word in enumerate(doc_doc_id[0])])
        return doc_words

    @classmethod
    def doc_word_counts(cls, docs, has_id=False):
        if not has_id:
            docs = docs.zipWithUniqueId()

        doc_word_counts = docs.flatMap(lambda doc_doc_id5: [(word, (doc_doc_id5[1], count))
                                                              for word, count in six.iteritems(Counter(doc_doc_id5[0]))])
        return doc_word_counts

    @classmethod
    def doc_frequency(cls, doc_word_counts):
        doc_frequency = doc_word_counts.map(lambda word_doc_id_count: (word_doc_id_count[0], 1)).reduceByKey(lambda x, y: x + y)
        return doc_frequency

    @classmethod
    def filter_min_doc_frequency(cls, doc_frequency, min_count=2):
        return doc_frequency.filter(lambda key_count: key_count[1] >= min_count)

    @classmethod
    def update_doc_frequency(cls, doc_frequency, batch_frequency):
        updated = doc_frequency.union(batch_frequency).reduceByKey(lambda x, y: x + y)
        return updated

    @classmethod
    def doc_scores(cls, doc_words, doc_frequency, total_docs, min_count=1):
        if min_count > 1:
            doc_frequency = cls.filter_min_doc_frequency(doc_frequency, min_count=min_count)

        num_partitions = doc_words.getNumPartitions()

        doc_ids_word_stats = doc_words.join(doc_frequency).map(lambda word_doc_id_pos_doc_frequency: (word_doc_id_pos_doc_frequency[0][0], (word_doc_id_pos_doc_frequency[0], word_doc_id_pos_doc_frequency[0][1], word_doc_id_pos_doc_frequency[1][1])))
        docs_tfidf = doc_ids_word_stats.groupByKey() \
                                       .mapValues(lambda vals: [(word, TFIDF.tfidf_score(1.0, doc_frequency, total_docs)) for word, pos, doc_frequency in sorted(vals, key=operator.itemgetter(1))])

        return docs_tfidf.coalesce(num_partitions)


class GeoTFIDFSpark(TFIDFSpark, GeoWordIndexSpark):
    @classmethod
    def doc_words(cls, docs, geo_aliases=None, has_id=False, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id6: doc_lat_lon_doc_id6[0][1] is not None and doc_lat_lon_doc_id6[0][2] is not None)

        if geo_aliases:
            doc_geohashes = docs.map(lambda doc_lat_lon_doc_id: (cls.geohash(doc_lat_lon_doc_id[0][1], doc_lat_lon_doc_id[0][2], geohash_precision=geohash_precision), (doc_lat_lon_doc_id[0][0], doc_lat_lon_doc_id[1]))) \
                                .leftOuterJoin(geo_aliases) \
                                .map(lambda geo_doc_doc_id_geo_alias: (geo_doc_doc_id_geo_alias[1][1] or geo_doc_doc_id_geo_alias[0], (geo_doc_doc_id_geo_alias[0][0], geo_doc_doc_id_geo_alias[0][1])))
        else:
            doc_geohashes = docs.map(lambda doc_lat_lon_doc_id1: (cls.geohash(doc_lat_lon_doc_id1[0][1], doc_lat_lon_doc_id1[0][2], geohash_precision=geohash_precision), (doc_lat_lon_doc_id1[0][0], doc_lat_lon_doc_id1[1])))

        doc_words = doc_geohashes.flatMap(lambda geo_doc_doc_id: [((geo_doc_doc_id[0], word), (geo_doc_doc_id[1][1], pos))
                                                                        for pos, word in enumerate(geo_doc_doc_id[1][0])])
        return doc_words

    @classmethod
    def doc_word_counts(cls, docs, geo_aliases=None, has_id=False, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id7: doc_lat_lon_doc_id7[0][1] is not None and doc_lat_lon_doc_id7[0][2] is not None)

        if geo_aliases:
            doc_geohashes = docs.flatMap(lambda doc_lat_lon_doc_id2: [(gh, (doc_lat_lon_doc_id2[0][0], doc_lat_lon_doc_id2[1])) for gh in cls.geohashes(doc_lat_lon_doc_id2[0][1], doc_lat_lon_doc_id2[0][2])]) \
                                .leftOuterJoin(geo_aliases) \
                                .map(lambda geo_doc_doc_id_geo_alias3: (geo_doc_doc_id_geo_alias3[1][1] or geo_doc_doc_id_geo_alias3[0], (geo_doc_doc_id_geo_alias3[0][0], geo_doc_doc_id_geo_alias3[0][1])))
        else:
            doc_geohashes = docs.flatMap(lambda doc_lat_lon_doc_id4: [(gh, (doc_lat_lon_doc_id4[0][0], doc_lat_lon_doc_id4[1])) for gh in cls.geohashes(doc_lat_lon_doc_id4[0][1], doc_lat_lon_doc_id4[0][2])])

        doc_word_counts = doc_geohashes.flatMap(lambda geo_doc_doc_id8: [((geo_doc_doc_id8[0], word), (geo_doc_doc_id8[1][1], count))
                                                                              for word, count in six.iteritems(Counter(geo_doc_doc_id8[1][0]))])
        return doc_word_counts

    @classmethod
    def filter_min_doc_frequency(cls, doc_frequency, min_count=2):
        return doc_frequency.filter(lambda key_count9: key_count9[1] >= min_count)

    @classmethod
    def total_docs_by_geo(cls, docs, has_id=False, geohash_precision=GeoWordIndexSpark.DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id10: doc_lat_lon_doc_id10[0][1] is not None and doc_lat_lon_doc_id10[0][2] is not None)

        total_docs_by_geo = docs.flatMap(lambda doc_lat_lon_doc_id11: [(gh, 1) for gh in cls.geohashes(doc_lat_lon_doc_id11[0][1], doc_lat_lon_doc_id11[0][2])]) \
                                .reduceByKey(lambda x, y: x + y)
        return total_docs_by_geo

    @classmethod
    def update_total_docs_by_geo(cls, total_docs_by_geo, batch_docs_by_geo):
        updated = total_docs_by_geo.union(batch_docs_by_geo).reduceByKey(lambda x, y: x + y)
        return updated

    @classmethod
    def updated_total_docs_geo_aliases(cls, total_docs_by_geo, geo_aliases):
        batch_docs_by_geo = total_docs_by_geo.join(geo_aliases) \
                                             .map(lambda geo_count_geo_alias: (geo_count_geo_alias[1][1], geo_count_geo_alias[1][0])) \
                                             .reduceByKey(lambda x, y: x + y)

        return cls.update_total_docs_by_geo(total_docs_by_geo, batch_docs_by_geo) \
                  .subtractByKey(geo_aliases)

    @classmethod
    def doc_scores(cls, doc_words, geo_doc_frequency, total_docs_by_geo, min_count=1):
        if min_count > 1:
            geo_doc_frequency = cls.filter_min_doc_frequency(geo_doc_frequency, min_count=min_count)

        num_partitions = doc_words.getNumPartitions()

        geo_doc_frequency_totals = geo_doc_frequency.map(lambda geo_word_count: (geo_word_count[0][0], (geo_word_count[0][1], geo_word_count[1]))) \
                                                    .join(total_docs_by_geo) \
                                                    .map(lambda geo_word_count_num_docs: ((geo_word_count_num_docs[0], geo_word_count_num_docs[0][0]), (geo_word_count_num_docs[0][1], geo_word_count_num_docs[1][1])))
        doc_ids_word_stats = doc_words.join(geo_doc_frequency_totals) \
                                      .map(lambda geo_word_doc_id_pos_doc_frequency_num_docs: (geo_word_doc_id_pos_doc_frequency_num_docs[0][0], (geo_word_doc_id_pos_doc_frequency_num_docs[0][1], geo_word_doc_id_pos_doc_frequency_num_docs[0][1], geo_word_doc_id_pos_doc_frequency_num_docs[1][0], geo_word_doc_id_pos_doc_frequency_num_docs[1][1])))

        docs_tfidf = doc_ids_word_stats.groupByKey() \
                                       .mapValues(lambda vals: [(word, TFIDF.tfidf_score(1.0, doc_frequency, num_docs)) for word, pos, doc_frequency, num_docs in sorted(vals, key=operator.itemgetter(1))])

        return docs_tfidf.coalesce(num_partitions)

    @classmethod
    def save(cls, word_info_gain, index_path):
        word_info_gain.map(lambda geo_word_val: '\t'.join([geo_word_val[0][0], geo_word_val[0][1], geo_word_val[1]])) \
                      .saveAsTextFile(index_path)
