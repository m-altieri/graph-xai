import numpy as np


def fidelity_score(true, pred, from_seqs=False, model=None):
    """Compute the fidelity score.
    This function can compute model fidelity or phenomenon fidelity depending on the arguments:
    - to compute model fidelity, pass the prediction on the original sequence and the prediction on the perturbed sequence;
    - to compute phenomenon fidelity, pass the ground truth and the prediction on the perturbed sequence.

    Args:
        true (if from_seqs, sequence of shape ([B,]T,N,F); otherwise, model prediction of shape ([B,]T,N)): if from_seqs, original sequence; otherwise, ground truth for the predicted sequence.
        pred (if from_seqs, sequence of shape ([B,],T,N,F); otherwise, model prediction of shape ([B,]T,N)): if from_seqs, perturbed sequence; otherwise, model prediction.
        from_seqs (bool, optional): if from_seqs, true and pred are sequences instead of predictions. In that case, the model parameter is required. Defaults to False.
        model (TensorFlow model, optional): if from_seqs, it must be a callable TF Model. Defaults to None.

    Raises:
        ValueError: if from_seqs is True, then model must not be None, and true and pred must have shape (T,N,F)
    """
    # ValueError checks
    if from_seqs and model is None:
        raise ValueError("if from_seqs is True, then model must not be None")
    if (
        true.ndim > 3 + from_seqs
        or true.ndim < 2 + from_seqs
        or pred.ndim > 3 + from_seqs
        or pred.ndim < 2 + from_seqs
        or true.ndim != pred.ndim
    ):
        raise ValueError(
            f"true and pred have the wrong number of dimensions, they must be ([B,]T,N,F) if from_seqs or ([B,]T,N) if not from_seqs, but have shapes {true.shape} and {pred.shape}"
        )

    has_batch_axis = true.ndim == (3 + from_seqs)

    if from_seqs:
        if not has_batch_axis:
            true = np.expand_dims(true, axis=0)
            pred = np.expand_dims(pred, axis=0)
        true = model.predict(true, verbose=0)
        pred = model.predict(pred, verbose=0)

    # Fidelity is computed using MAE
    mae = np.mean(np.abs(true - pred))
    return mae


def fidelity_score_rf(true, pred, from_seqs=False, model=None):
    has_batch_axis = true.ndim == (3 + from_seqs)

    if from_seqs:
        if not has_batch_axis:
            true = np.expand_dims(true, axis=0)
            pred = np.expand_dims(pred, axis=0)
        B, T, N, F = true.shape
        # o (b,t,n,f) o (b,t,n)
        true = np.reshape(true, (B * T * N, F))
        pred = np.reshape(pred, (B * T * N, F))
        true = model.predict(true)
        pred = model.predict(pred)

    # Fidelity is computed using MAE
    mae = np.mean(np.abs(true - pred))
    return mae
