from scipy.special import comb 
import  numpy  as np 


def ensemble_error(n_classifier, error):

    """Calculate the ensemble error of a majority vote from 
    n_classifier base classifiers with an individual error of error.
    
    Parameters
    ----------
    n_classifier : int
        The number of classifiers in the ensemble.
    error : float
        The error of an individual classifier.
        
    Returns
    -------
    ensemble_error : float
        The ensemble error of the majority vote from n_classifier base classifiers.
        
    """
    
    k_start = int(np.ceil(n_classifier / 2.))
    
    ensemble_error = 0.0
    
    for k in range(k_start, n_classifier + 1):
        ensemble_error += comb(n_classifier, k) * error**k * (1 - error)**(n_classifier - k)
        
    return ensemble_error


example_error = ensemble_error(n_classifier=11, error=0.25)
print('Ensemble error: {:.3f}'.format(example_error))
