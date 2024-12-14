import torch as th
import numpy as np

def masked_spectral_distance(y_true, y_pred):
    """
    Calculates the masked spectral distance between true and predicted intensity vectors.
    The masked spectral distance is a metric for comparing the similarity between two intensity vectors.

    Masked, normalized spectral angles between true and pred vectors

    > arccos(1*1 + 0*0) = 0 -> SL = 0 -> high correlation

    > arccos(0*1 + 1*0) = pi/2 -> SL = 1 -> low correlation

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : tf.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    tf.Tensor
        A tensor containing the masked spectral distance between `y_true` and `y_pred`.

    """

    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = 1e-9

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 norm
    pred_norm = true_masked / true_masked.norm(dim=1, keepdim=True)
    true_norm = pred_masked / pred_masked.norm(dim=1, keepdim=True)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = (pred_norm * true_norm).sum(1)
    arccos = th.acos(product)
    return 2 * arccos / np.pi

def masked_pearson_correlation_distance(y_true, y_pred):
    """
    Calculates the masked Pearson correlation distance between true and predicted intensity vectors.
    The masked Pearson correlation distance is a metric for comparing the similarity between two intensity vectors,
    taking into account only the non-negative values in the true values tensor (which represent valid peaks).

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : tf.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    tf.Tensor
        A tensor containing the masked Pearson correlation distance between `y_true` and `y_pred`.

    """

    epsilon = 1e-9

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    mx = true_masked.mean(1, keepdim=True) #tf.math.reduce_mean(true_masked)
    my = pred_masked.mean(1, keepdim=True) #tf.math.reduce_mean(pred_masked)
    xm, ym = true_masked - mx, pred_masked - my
    r_num = (xm * ym).mean(1, keepdim=False) #tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = xm.std(1, keepdim=False) * ym.std(1, keepdim=False) #tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return 1 - (r_num / r_den)

def cosine_score(y_true, y_pred, sqrt=True):
    # correct loader setting -1 for impossible ions
    mask = y_true < 0
    y_true[mask] = 0
    if sqrt:
        y_true = th.sqrt(y_true)
        y_pred = th.sqrt(y_pred)
    return -th.nn.functional.cosine_similarity(y_true, y_pred)
