def expand_query(query, synonym_map):
    expanded = [query]
    for term, synonyms in synonym_map.items():
        if term.lower() in query.lower():
            expanded.extend(synonyms)
    return list(set(expanded))
