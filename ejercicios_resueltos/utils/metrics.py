import numpy as np


class BaseMetric:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def __call__(self, *args, **kwargs):
        pass


class Precision(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        prediction = self.parameters["predictions"]
        truth = self.parameters["truth"]
        
        TP_mask = (truth == 1) & (prediction == 1)
        FP_mask = (truth == 0) & (prediction == 1)
        
        TP = TP_mask.sum()
        FP = FP_mask.sum()
        
        return TP / (TP + FP)


class Recall(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        prediction = self.parameters["predictions"]
        truth = self.parameters["truth"]

        TP_mask = (truth == 1) & (prediction == 1)
        FN_mask = (truth == 1) & (prediction == 0)
        
        TP = TP_mask.sum()
        FN = FN_mask.sum()
        
        return TP / (TP + FN)


class Accuracy(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        prediction = self.parameters["predictions"]
        truth = self.parameters["truth"]
        
        TP_mask = (truth == 1) & (prediction == 1)
        FP_mask = (truth == 0) & (prediction == 1)
        FN_mask = (truth == 1) & (prediction == 0)
        TN_mask = (truth == 0) & (prediction == 1)

        TP = TP_mask.sum()
        FP = FP_mask.sum()
        FN = FN_mask.sum()
        TN = TN_mask.sum()

        return (TP + TN) / (TP + TN + FP + FN)


class AvgQPrecision(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):
        query_ids = self.parameters["query_ids"]
        truth_relevance = self.parameters["truth_relevance"]

        true_relevance_mask = (truth_relevance == 1) 

        filtered_query_id = query_ids[true_relevance_mask] 
        
        filtered_true_relevance_count = np.bincount(filtered_query_id) 

        # contar queries con 0 en queries sin documentos relevantes
        unique_query_ids = np.unique(query_ids)                             
        non_zero_count_idxs = np.where(filtered_true_relevance_count > 0)   
        true_relevance_count = np.zeros(unique_query_ids.max() + 1)         
        
        true_relevance_count[non_zero_count_idxs] = filtered_true_relevance_count[non_zero_count_idxs] 

        true_relevance_count_by_query = true_relevance_count[unique_query_ids] 

        fetched_documents_count = np.bincount(query_ids)[unique_query_ids] 

        precision_by_query = true_relevance_count_by_query / fetched_documents_count 

        return np.mean(precision_by_query) 


class MSE(BaseMetric):
    def __init__(self):
        BaseMetric.__init__(self)

    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n
    
class R2_lr(BaseMetric):
    def __init__(self):
        BaseMetric.__init__(self)

    def __call__(self, X, y, y_hat):
        n = y.size
        var_media = np.sum((y - np.mean(y)) ** 2) / n
        var_fit = np.sum((y - y_hat) ** 2) / n
        return (var_media - var_fit)/var_media
