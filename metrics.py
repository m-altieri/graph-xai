import tensorflow as tf


def avg_sub(x1, x2):
    return tf.reduce_mean(tf.math.abs(tf.math.subtract(x1, x2)))


class Metric:
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.total = tf.Variable(0.0)
        self.count = tf.Variable(0.0)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        raise NotImplementedError()

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)


class NormalizedModelFidelityPlus(Metric):
    def __init__(self, name="nMF+"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        all_zeros = tf.zeros_like(historical)
        model = kwargs.get("model")
        pred_on_zeros = model(all_zeros)

        mfp = avg_sub(pred_on_nonrelevant, pred_on_original)
        mfp_zeros = avg_sub(pred_on_zeros, pred_on_original)
        mfp_original = avg_sub(pred_on_original, pred_on_original)
        normalized_fidelity = avg_sub(mfp, mfp_original) / avg_sub(
            mfp_zeros, mfp_original
        )

        self.total.assign_add(normalized_fidelity)
        self.count.assign_add(1.0)


class NormalizedModelFidelityMinus(Metric):
    def __init__(self, name="nMF-"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        all_zeros = tf.zeros_like(historical)
        model = kwargs.get("model")
        pred_on_zeros = model(all_zeros)

        mfm = avg_sub(pred_on_relevant, pred_on_original)
        mfm_zeros = avg_sub(pred_on_zeros, pred_on_original)
        mfm_original = avg_sub(pred_on_original, pred_on_original)
        normalized_fidelity = avg_sub(mfm, mfm_original) / avg_sub(
            mfm_zeros, mfm_original
        )

        self.total.assign_add(normalized_fidelity)
        self.count.assign_add(1.0)


class NormalizedPhenomenonFidelityPlus(Metric):
    def __init__(self, name="nPF+"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        all_zeros = tf.zeros_like(historical)
        model = kwargs.get("model")
        pred_on_zeros = model(all_zeros)

        error_on_original = avg_sub(phenomenon[..., 0], pred_on_original)
        error_on_nonrelevant = avg_sub(phenomenon[..., 0], pred_on_nonrelevant)
        error_on_zeros = avg_sub(phenomenon[..., 0], pred_on_zeros)

        pfp = avg_sub(error_on_original, error_on_nonrelevant)
        pfp_zeros = avg_sub(error_on_original, error_on_zeros)
        pfp_original = avg_sub(error_on_original, error_on_original)
        normalized_fidelity = avg_sub(pfp, pfp_original) / avg_sub(
            pfp_zeros, pfp_original
        )

        self.total.assign_add(normalized_fidelity)
        self.count.assign_add(1.0)


class NormalizedPhenomenonFidelityMinus(Metric):
    def __init__(self, name="nPF-"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        all_zeros = tf.zeros_like(historical)
        model = kwargs.get("model")
        pred_on_zeros = model(all_zeros)

        error_on_original = avg_sub(phenomenon[..., 0], pred_on_original)
        error_on_relevant = avg_sub(phenomenon[..., 0], pred_on_relevant)
        error_on_zeros = avg_sub(phenomenon[..., 0], pred_on_zeros)

        pfm = avg_sub(error_on_original, error_on_relevant)
        pfm_zeros = avg_sub(error_on_original, error_on_zeros)
        pfm_original = avg_sub(error_on_original, error_on_original)
        normalized_fidelity = avg_sub(pfm, pfm_original) / avg_sub(
            pfm_zeros, pfm_original
        )

        self.total.assign_add(normalized_fidelity)
        self.count.assign_add(1.0)


class ModelFidelityPlus(Metric):
    def __init__(self, name="MF+"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        fidelity = tf.reduce_mean(
            tf.abs(tf.math.subtract(pred_on_original, pred_on_nonrelevant))
        )
        self.total.assign_add(fidelity)
        self.count.assign_add(1.0)


class ModelFidelityMinus(Metric):
    def __init__(self, name="MF-"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        fidelity = tf.reduce_mean(
            tf.abs(tf.math.subtract(pred_on_original, pred_on_relevant))
        )
        self.total.assign_add(fidelity)
        self.count.assign_add(1.0)


class PhenomenonFidelityPlus(Metric):
    def __init__(self, name="PF+"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        fidelity = tf.reduce_mean(
            tf.abs(
                tf.math.subtract(
                    tf.abs(tf.math.subtract(phenomenon[..., 0], pred_on_original)),
                    tf.abs(tf.math.subtract(phenomenon[..., 0], pred_on_nonrelevant)),
                )
            )
        )
        self.total.assign_add(fidelity)
        self.count.assign_add(1.0)


class PhenomenonFidelityMinus(Metric):
    def __init__(self, name="PF-"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        fidelity = tf.reduce_mean(
            tf.abs(
                tf.math.subtract(
                    tf.abs(tf.math.subtract(phenomenon[..., 0], pred_on_original)),
                    tf.abs(tf.math.subtract(phenomenon[..., 0], pred_on_relevant)),
                )
            )
        )
        self.total.assign_add(fidelity)
        self.count.assign_add(1.0)


class Sparsity(Metric):
    def __init__(self, name="S"):
        super().__init__(name)

    def update_state(
        self,
        historical,
        phenomenon,
        pred_on_original,
        relevant_mask,
        pred_on_relevant,
        nonrelevant_mask,
        pred_on_nonrelevant,
        **kwargs,
    ):
        sparsity_fn = lambda x: 1.0 - tf.cast(tf.reduce_sum(x), tf.float32) / tf.cast(
            tf.size(x), tf.float32
        )
        self.total.assign_add(sparsity_fn(relevant_mask))
        self.count.assign_add(1.0)
