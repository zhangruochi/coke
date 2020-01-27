import numpy as np

def get_all_topic_words(H, num_top_words = 10):
    """ get topic words through matrix factorization
    
    Parameters
    ----------
    H (np.matrix) : 
    num_top_words (int) : number of words in specific topic
        
    Returns
    -------
    list of topics
    """
    top_indices = lambda t: {i for i in np.argsort(t)[:-num_top_words-1:-1]}
    topic_indices = [top_indices(t) for t in H]
    return sorted(set.union(*topic_indices))

# U, s, Vh = linalg.svd(vectors, full_matrices=False)
# get_all_topic_words(vh[:10],num_top_words = 10)
