import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def create_similarity_vector(reference_feature_vector,query_features):
    # query features must be a 2-D vector: np.array([[X,X...]])
    # reference features must be a 2D array
    similarity_vector = cdist(reference_feature_vector,query_features,'euclidean')
    return similarity_vector

def create_similarity_matrix(reference_feature_vector,query_feature_vector):
    # query features must be a 2D array of feature vectors: np.array([[X,X...],[X,X...]])
    # reference features must be a 2D array of feature vectors
    similarity_matrix = cdist(reference_feature_vector,query_feature_vector,'euclidean')
    return similarity_matrix

def create_normalised_similarity_matrix(reference_feature_vector,query_feature_vector):
    # query features must be a 2D array of feature vectors: np.array([[X,X...],[X,X...]])
    # reference features must be a 2D array of feature vectors
    Sref = cdist(reference_feature_vector,reference_feature_vector,'euclidean')
    S = cdist(reference_feature_vector,query_feature_vector,'euclidean')
    Snorm = (S - Sref.mean())/Sref.std()
    return Snorm, Sref.mean(), Sref.std()

def find_best_match(similarity_matrix):
    # works for similarity vectors as well as matrices
    # TODO: need to identify if there are multiple entries with the best match!
    return np.argmin(similarity_matrix,axis=0)

def find_next_best_match(similarity_matrix):
    S_sorted = np.argsort(similarity_matrix,axis=0)  #create matrix with each row annotated with the minimum order
    return np.argmin(abs(S_sorted-1),axis=0)         #return the index values with a 1 (the second minimum value in each column)

def find_nth_best_match(similarity_matrix, n):       #n is 2 for 2nd, 3 for 3rd, etc
    if type(n) != int:
        print('ERROR find_nth_best_match: n must be an integer')
        return
    if n > similarity_matrix.shape[0]:
        print('WARNING find_nth_best_match: n is larger than number of reference images, changing n to max size')
        n = similarity_matrix.shape[0]
    if n == 0:
        print('WARNING find_nth_best_match: n should be larger than zero (assume you mean n=0 for best match)')
        n = 1
    S_sorted = np.argsort(similarity_matrix,axis=0)  #create matrix with each row annotated with the minimum order
    return np.argmin(abs(S_sorted-(n-1)),axis=0)     #return the nth index values

def find_best_match_distances(similarity_matrix):
    # works for similarity vectors as well as matrices
    return np.min(similarity_matrix,axis=0)

def find_actual_match_distances(similarity_matrix, actual_match_idx):
    number_queries = len(actual_match_idx)
    act_match_dist = np.full(number_queries,np.nan)
    for query_number,match_index in enumerate(actual_match_idx):
        act_match_dist[query_number]=similarity_matrix[match_index,query_number]
    return act_match_dist    

def find_nth_best_match_distances(similarity_matrix,n):
    if type(n) != int:
        print('ERROR find_nth_best_match_distances: n must be an integer')
        return
    if n > similarity_matrix.shape[0]:
        print('WARNING find_nth_best_match_distances: n is larger than number of reference images, changing n to max size')
        n = similarity_matrix.shape[0]
    if n == 0:
        print('WARNING find_nth_best_match_distances: n should be larger than zero (assume you mean n=0 for best match)')
        n = 1
    S_sorted = np.sort(similarity_matrix,axis=0)  #create matrix with each row annotated with the minimum order
    return S_sorted[n-1,:] 

def apply_distance_threshold(best_match_distances, threshold):
    return best_match_distances < threshold

def extract_similarity_vector(Smatrix, query_number):
    if type(query_number) != int:
        print('ERROR extract_similarity_vector: query number must be an integer')
        return
    if (query_number >= Smatrix.shape[0]) | (query_number < 0):
        print('ERROR extract_similarity_vector: query number is less than zero or larger than query vector')
        return
    return Smatrix[:,query_number]

def find_frame_error(S,ground_truth):
    best_match = find_best_match(S)
    return abs(ground_truth-best_match)

def is_within_frame_tolerance(frame_error,tolerance): #frame error as np.array
    if not issubclass(frame_error.dtype.type, np.integer):
        print('ERROR is_within_frame_tolerance: frame_error does not contain integers')
        return
    return(frame_error <= tolerance)

def is_within_position_error(position_error,tolerance): #position error as np.array
    return(position_error <= tolerance)

def find_metrics(tp,fp,tn,fn,verbose):
    num_tp=sum(tp)
    num_fp=sum(fp)
    num_tn=sum(tn)
    num_fn=sum(fn)
    precision=num_tp/(num_tp+num_fp)
    recall=num_tp/(num_tp+num_fn)
    if verbose == True:
        print('TP={0}, TN={1}, FP={2}, FN={3}'.format(num_tp,num_tn,num_fp,num_fn))
        print('precision={0:3.1f}%  recall={1:3.1f}%\n'.format(precision*100,recall*100))
    return [precision, recall, num_tp, num_fp, num_tn, num_fn]

def find_vpr_performance_metrics(match_found,in_tolerance,match_exists,verbose=False): #as np.array

    if ((not issubclass(match_found.dtype.type, np.bool_)) or 
         (not issubclass(in_tolerance.dtype.type, np.bool_)) or
          (not issubclass(match_exists.dtype.type, np.bool_))):
        print('ERROR find_vpr_performance_metrics: inputs must all be boolean arrays')
        return

    no_match_found = np.logical_not(match_found).astype('int')
    not_in_tolerance = np.logical_not(in_tolerance).astype('int')
    no_match_exists = np.logical_not(match_exists).astype('int')

    tp = in_tolerance & match_found & match_exists
    tn = no_match_found & no_match_exists
    fn = match_exists & no_match_found
    fp = match_found & (not_in_tolerance | no_match_exists)

    return find_metrics(tp,fp,tn,fn,verbose)

def find_prediction_performance_metrics(predicted_in_tolerance,actually_in_tolerance,verbose=False): #as np.array
    
    predicted_in_tolerance=predicted_in_tolerance.astype('bool')
    actually_in_tolerance=actually_in_tolerance.astype('bool')
    
    tp = predicted_in_tolerance & actually_in_tolerance
    fp = predicted_in_tolerance & ~actually_in_tolerance
    tn = ~predicted_in_tolerance & ~actually_in_tolerance
    fn = ~predicted_in_tolerance & actually_in_tolerance
    return find_metrics(tp,fp,tn,fn,verbose)

def find_y(S,actual_match,tolerance_threshold):
    frame_error = find_frame_error(S,actual_match)
    in_tol = is_within_frame_tolerance(frame_error,tolerance_threshold)
    return in_tol

def find_closedloop_performance_metrics(S,actual_match,tolerance,pred_intol):

    actually_intol=find_y(S,actual_match,tolerance)
    match_exists = np.full_like(pred_intol,True)  # Assume that a match exist for each query

    match_distances=find_best_match_distances(S)
    d_sweep = np.linspace(match_distances.min(),match_distances.max(),2000) #2000 is number of points in p-r curve
    
    p=np.full_like(d_sweep,np.nan)
    r=np.full_like(d_sweep,np.nan)
    tp=np.full_like(d_sweep,np.nan)
    fp=np.full_like(d_sweep,np.nan)
    tn=np.full_like(d_sweep,np.nan)
    fn=np.full_like(d_sweep,np.nan)

    for i, v in enumerate(d_sweep):
        match_found = apply_distance_threshold(match_distances, v) & pred_intol.astype('bool')
        [p[i], r[i], tp[i], fp[i], tn[i], fn[i]] = find_vpr_performance_metrics(match_found,actually_intol,match_exists,verbose=False)
    return p,r,tp,fp,tn,fn,d_sweep

def find_baseline_performance_metrics(S,actual_match,tolerance):
    return find_closedloop_performance_metrics(S,actual_match,tolerance, np.full(len(actual_match),True))

def plot_baseline_vs_closedloop_PRcurves(S,actual_match,tolerance,y_pred):
    p,r,tp,fp,tn,fn,d=find_baseline_performance_metrics(S,actual_match,tolerance)
    plt.plot(r,p);
    p,r,tp,fp,tn,fn,d=find_closedloop_performance_metrics(S,actual_match,tolerance,y_pred)
    plt.plot(r,p); 
    plt.xlim(0,1); plt.ylim(0,1);
    plt.ylabel('Precision');
    plt.xlabel('Recall');
    plt.legend(['baseline','closed-loop']);
    plt.title('Baseline vs Closed-Loop Performance');
    return

def find_mean_localisation_error(S,actual_match,y_pred=[0]):
    if len(y_pred)==1: # default case where y_pred is not specified
        y_pred=np.full(len(actual_match),True)
    return find_frame_error(S,actual_match)[y_pred].mean()