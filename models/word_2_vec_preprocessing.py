import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import re
from gensim.models.word2vec import Word2Vec
import string
from typing import Union


############################################################################

REGEX_URL = r'((http|https)\:\/\/)[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
clear_url = lambda text: re.sub(REGEX_URL, ' ', text)
DOT_REGEX = r"(?<!\w)(?:[A-Z][A-Za-z]{,3}|[a-z]{1,2})\."

############################################################################

@dataclass(frozen=True)
class Preprocessing:
    """Preprocessing class used to preprocess news text before Text
    Summarization is applied.
    
    - Usage:
    ```
    >>> preprocessor = Preprocessing()
    >>> text = "any news text"
    >>> site_name = "media site"
    >>> clean_text = preprocessor(text, site_name)
    ```
    """

    def _clear_content_head(self, content: str, site_name: str,
                           head_pattern: str=r"\s\-+\s") -> str:
        """used to clear any head in given news content"""

        match = re.search(head_pattern, content)
        if match:
            idx_end = match.end()
            site_name = site_name.split()[0]
            if site_name.lower() in content[:idx_end].lower():
                content = content[idx_end:]

        return content


#################################

    def _clear_abbreviation_dot(self, text: str) -> str:
        """used to rip off abbreviation dot in given text"""

        # replace any matched abbr with empty string
        text_list = list(text)
        for i, match in enumerate(re.finditer(DOT_REGEX, text)):
            no_dot = match.group().replace('.', '')
            idx = match.span()
            text_list[idx[0]-i: idx[1]-i] = no_dot

        # join list text and clear multiple whitespaces
        text = ''.join(text_list)
        text = re.sub(' +', ' ', text)
    
#################################

    def __call__(self, content: str, site_name: str) -> Union[str, bool]:

        """the method is used to:
        - clear any content head
        - clear any heading/tailing whitespace & punct
        - clear any abbreviation dot
        Args:
        - content (str): news content
        - site_name (str): news site name
        Return:
        preprocessed content
        """

        content = self._clear_content_head(content, site_name)
        content = clear_url(content)

        # clear leadding/trailing whitespaces & puncts
        content = content.strip(string.punctuation)
        content = content.strip()

        # change multiple whitespaces to single one
        content = re.sub(' +', ' ', content)

        # clear whitespace before dot
        content = re.sub(r'\s+([?,.!"])', r'\1', content)

        return content
    

@dataclass(frozen=True)
class Embedder:
    """This class is used to create word embeddings from given sentence.
    The processes implemented are the following:
    - convert each token of given sentence to its representative vector;
    - calculate mean of all tokens in given sentence in order to get a
    sentence embedding.
    Arg:
    - model: a gensim Word2Vec model
    """

    model: Word2Vec

######################

    def __get_vector(self, token: str) -> np.ndarray:
        """used to convert given token to its representative vector"""
        try:
            return self.model.wv.get_vector(token)
        except KeyError:
            return False

######################

    def __averaging(self, token_matrix: np.ndarray) -> np.ndarray:
        """used to calculate mean of an array of vectors in order to get a
        sentence embedding"""
        return np.mean(token_matrix, axis=0)

######################

    def embed(self, sentence: str, return_oov: bool=False) -> np.ndarray:
        """combine all other methods to execute the embedding process.
        
        Args:
        - sentence (str): a sentence to be process to get its embedding
        - return_oov(bool): indicate if you'd like to return the OOV
        (out-of-vocabulary) tokens
        
        Returns:
        If all tokens in given sentence are OOV tokens, return False (and with
        list of OOVs if 'return_oov' set to True).
        else, return the sentence embedding (and with list of OOVs if
        'return_oov' set to True).
        """

        # make the given sentence lower and collect only words
        list_tok = re.findall(r"\w+", sentence.lower())

        # buffers
        list_vec = []
        OOV_tokens = []

        # loop through each token of given sentence
        for token in list_tok:
            tokvec = self.__get_vector(token) # convert to vector

            # check if no OOV token produced
            if isinstance(tokvec, np.ndarray):
                list_vec.append(tokvec)
            else:
                OOV_tokens.append(token)

        # if all tokens in given sentence are OOV tokens
        if not list_vec:
            if return_oov:
                return False, OOV_tokens
            return False

        # if not
        list_vec = np.array(list_vec)
        if return_oov:
            return (self.__averaging(list_vec), OOV_tokens)
        return self.__averaging(list_vec)
    

##################################################################

@dataclass(frozen=True)
class Clustering:
    """This class is used to cluster sentence embeddings in order to execute
    text summarization. The processes implemented are thr following:
    - define a KNN clustering model;
    - train the model;
    - find sentences closest to the cluster's center.
    Args:
    - features (np.ndarray): sentence embeddings
    - random_state (int - optional): random state for random seed
    """

    features: np.ndarray
    random_state: int = 1

######################

    def __define_model(self, k: int) -> None:
        """used to define KNN clustering model"""

        model = KMeans(n_clusters=k, random_state=self.random_state)
        object.__setattr__(self, 'model', model)

######################

    def __find_closest_sents(self, centroids: np.ndarray) -> Dict:
        """
        Find the closest arguments to centroid.
        - centroids: Centroids to find closest.
        - return: Closest arguments.
        """

        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []

        for j, centroid in enumerate(centroids):

            for i, feature in enumerate(self.features):
                value = np.linalg.norm(feature - centroid)

                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        return args

######################

    def cluster(self, ratio: float = 0.2,
                num_sentences: int = None) -> List[int]:
        """
        Clusters sentences based on the ratio.
        - ratio: Ratio to use for clustering.
        - num_sentences: Number of sentences. Overrides ratio.
        return: Sentences index that qualify for summary.
        """

        # set k value
        if num_sentences is not None:
            if num_sentences == 0:
                return []
            k = min(num_sentences, len(self.features))
        else:
            k = max(int(len(self.features) * ratio), 1)

        # define n train the model
        self.__define_model(k)
        self.model.fit(self.features)

        # find the closest embeddings to the center
        centroids = self.model.cluster_centers_
        cluster_args = self.__find_closest_sents(centroids)

        sorted_values = sorted(cluster_args.values())
        return sorted_values

@dataclass(frozen=True)
class Word2VecSummarizer:
    """The main class for Word2Vec Summarizer
    Args:
    - model: A gensim Word2Vec model (optional)
    - random_state: state for random seed (optional)
    """
    def __init__(self, model: Word2Vec, random_state: int=1):
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'random_state', random_state)

######################

    def __split_sentence(self, text: str) -> List[str]:
        """used to split given text into sentences"""
        sentences = sent_tokenize(text)
        return [sent for sent in sentences if len(sent) >= 5]

######################

    def __set_embedder(self) -> None:
        """used to instantiate Embedder object"""
        embedder = Embedder(self.model)
        object.__setattr__(self, 'embedder', embedder)

######################

    def __set_clusterer(self, features: np.ndarray,
                        random_state: int) -> None:
        """used to instantiate Clustering object"""
        clusterer = Clustering(features, random_state)
        object.__setattr__(self, 'clusterer', clusterer)

######################

    def summarize(self, text: str,
                  use_first: bool = True,
                  num_sentences: int = None,
                  ratio: float = 0.2,
                  return_oov: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        This method executes the summarization part.
        
        Args:
        - text (str): text to be processed
        - use_first (bool-default True): indicate if the first sentence of the text used
        - num_sentences (int): whether you'd like to return certain number of summarized sentences (optional)
        - ratio (float-default 0.2): ratio of sentences to use
        - return_oov(bool-default False): indicate if you'd like to return the OOV
        (out-of-vocabulary) tokens
        
        Returns: tuple of sentences and related embeddings (and OOV list if return_oov set to True)
        """
        list_sentence = self.__split_sentence(text)
        self.__set_embedder()

        # set buffers
        sent_vecs = []
        oov_list = []

        # loop through each sentence to create each embeddings
        for sentence in list_sentence:
            if return_oov:
                vec, oov = self.embedder.embed(sentence, return_oov)
                oov_list.extend(oov)
            else:
                vec = self.embedder.embed(sentence, return_oov)

            # check if no OOV returned
            if isinstance(vec, np.ndarray):
                sent_vecs.append(vec)

        sent_vecs = np.array(sent_vecs) # create array of all embeddings

        # instantiate clustering & process
        self.__set_clusterer(sent_vecs, self.random_state)
        summary_idx = self.clusterer.cluster(ratio, num_sentences)

        if use_first:
            if not summary_idx:
                summary_idx.append(0)

            elif summary_idx[0] != 0:
                summary_idx.insert(0, 0)

        sentences = [list_sentence[idx] for idx in summary_idx]
        embeddings = np.asarray([sent_vecs[idx] for idx in summary_idx])

        if return_oov:
            return sentences, oov_list
        return sentences

