import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

def relevance_feedback(vec_docs, vec_queries, sim, n=10, alpha=0.75, beta=0.15, iterations=4):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    updated_vec_queries = vec_queries.copy().toarray()
    vec_docs_array = vec_docs.toarray()
    rf_sim = sim.copy()
    for iter in range(iterations):
        for i in range(sim.shape[1]):
            ranked_documents = np.argsort(-sim[:, i])[:n]
            not_n = sim.shape[0] - n
            for j in range(sim.shape[0]):
                if j in ranked_documents:
                    updated_vec_queries[i] += (alpha * vec_docs_array[j] / n)
                else:
                    updated_vec_queries[i] -= (beta * vec_docs_array[j] / not_n)
        rf_sim = cosine_similarity(vec_docs, updated_vec_queries)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n1=10, n2=5, alpha=0.75, beta=0.15, iterations=4):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    updated_vec_queries = vec_queries.copy().toarray()
    vec_docs_array = vec_docs.toarray()
    rf_sim = sim.copy()
    for iter in range(iterations):
        for i in range(rf_sim.shape[1]):
            ranked_documents = np.argsort(-rf_sim[:, i])[:n1]
            not_n1 = rf_sim.shape[0] - n1
            for j in range(rf_sim.shape[0]):
                if j in ranked_documents:
                    updated_vec_queries[i] += (alpha * vec_docs_array[j] / n1)
                else:
                    updated_vec_queries[i] -= (beta * vec_docs_array[j] / not_n1)
    vec_docs_array_normalized = Normalizer().fit_transform(vec_docs_array)
    term_sim = vec_docs_array_normalized.T.dot(vec_docs_array_normalized)
    for i in range(updated_vec_queries.shape[0]):
        most_relevant_term = np.argmax(updated_vec_queries[i])
        similar_terms = np.argsort(-term_sim[most_relevant_term])[1:n2+1]
        for term in similar_terms:
            # updated_vec_queries[i, term] = updated_vec_queries[i, most_relevant_term]
            updated_vec_queries[i, term] += updated_vec_queries[i, most_relevant_term]
            updated_vec_queries[i, term] /= 2
    rf_sim = cosine_similarity(vec_docs, updated_vec_queries)
    return rf_sim