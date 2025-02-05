import geohash
import operator


class GeoWordIndexSpark(object):
    DEFAULT_GEOHASH_PRECISION = 4

    @classmethod
    def geohash(cls, lat, lon, geohash_precision=DEFAULT_GEOHASH_PRECISION):
        return geohash.encode(lat, lon)[:geohash_precision]

    @classmethod
    def geohashes(cls, lat, lon, geohash_precision=DEFAULT_GEOHASH_PRECISION):
        gh = cls.geohash(lat, lon, geohash_precision=geohash_precision)
        return [gh] + geohash.neighbors(gh)

    @classmethod
    def geo_aliases(cls, total_docs_by_geo, min_doc_count=1000):
        keep_geos = total_docs_by_geo.filter(lambda geo_count: geo_count[1] >= min_doc_count)
        alias_geos = total_docs_by_geo.subtract(keep_geos)
        return list(alias_geos.keys()) \
                         .flatMap(lambda key: [(neighbor, key) for neighbor in geohash.neighbors(key)]) \
                         .join(keep_geos) \
                         .map(lambda neighbor_key_count: (neighbor_key_count[1][0], (neighbor_key_count[0], neighbor_key_count[1][1]))) \
                         .groupByKey() \
                         .map(lambda key_values: (key_values[0], sorted(key_values[1], key=operator.itemgetter(1), reverse=True)[0][0]))

    @classmethod
    def total_docs_by_geo(cls, docs, has_id=False, geohash_precision=DEFAULT_GEOHASH_PRECISION):
        if not has_id:
            docs = docs.zipWithUniqueId()

        docs = docs.filter(lambda doc_lat_lon_doc_id: doc_lat_lon_doc_id[0][1] is not None and doc_lat_lon_doc_id[0][2] is not None)

        total_docs_by_geo = docs.flatMap(lambda doc_lat_lon_doc_id1: [(gh, 1) for gh in cls.geohashes(doc_lat_lon_doc_id1[0][1], doc_lat_lon_doc_id1[0][2])]) \
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
