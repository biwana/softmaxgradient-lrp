import numpy as np
from keras import backend as K
from innvestigate.analyzer import BoundedDeepTaylor, GuidedBackprop
from innvestigate.analyzer import LRPEpsilon, LRPAlpha2Beta1, LRPSequentialPresetAFlat, LRPSequentialPresetBFlat, LRPAlphaBeta
from innvestigate.analyzer import DeepLIFT
from innvestigate.analyzer import Deconvnet, IntegratedGradients, SmoothGrad, InputTimesGradient
import innvestigate.utils as iutils
from keras.layers import Lambda, Layer
from skimage.transform import resize


class _SoftMax(Layer):
    def call(self, x):
        return [K.softmax(tmp) for tmp in iutils.to_list(x)]
    
class _Sigmoid(Layer):
    def call(self, x):
        return [K.sigmoid(tmp) for tmp in iutils.to_list(x)]
    
class LRPA2B1(LRPAlphaBeta):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(LRPA2B1, self).__init__(model, alpha=2, beta=1, **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
#         initial_R = Lambda(lambda x: (x * self.R_mask))(X)
#         return initial_R
        return X
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(LRPA2B1, self).analyze(inputs), 0)
        else:
            return super(LRPA2B1, self).analyze(inputs)
    
class Decon(Deconvnet):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(Decon, self).__init__(model, **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(Decon, self).analyze(inputs), 0)
        else:
            return super(Decon, self).analyze(inputs)
    
class IxG(InputTimesGradient):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(IxG, self).__init__(model, **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
#         initial_R = Lambda(lambda x: (x * self.R_mask))(X)
#         return initial_R
        return X
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(IxG, self).analyze(inputs), 0)
        else:
            return super(IxG, self).analyze(inputs)
    
class SG(SmoothGrad):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(SG, self).__init__(model, **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(SG, self).analyze(inputs), 0)
        else:
            return super(SG, self).analyze(inputs)
    
class IG(IntegratedGradients):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(IG, self).__init__(model, **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(IG, self).analyze(inputs), 0)
        else:
            return super(IG, self).analyze(inputs)
    
class DL(DeepLIFT):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(DL, self).__init__(model, **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
#         initial_R = Lambda(lambda x: (x * self.R_mask))(X)
#         return initial_R
        return X
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(DL, self).analyze(inputs), 0)
        else:
            return super(DL, self).analyze(inputs)
    
class LRPE(LRPEpsilon):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(LRPE, self).__init__(model, neuron_selection_mode="all", **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(LRPE, self).analyze(inputs), 0)
        else:
            return super(LRPE, self).analyze(inputs)
    
class LRPB(LRPSequentialPresetBFlat):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(LRPB, self).__init__(model, neuron_selection_mode="all", **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(LRPB, self).analyze(inputs), 0)
        else:
            return super(LRPB, self).analyze(inputs)

class LRPA(LRPSequentialPresetAFlat):
    def __init__(self, 
                 model,
                 target_id, 
                 relu=False, 
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        self.initialize_r_mask(R_mask)
        super(LRPA, self).__init__(model, neuron_selection_mode="all", **kwargs)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(LRPA, self).analyze(inputs), 0)
        else:
            return super(LRPA, self).analyze(inputs)

    
class _MaskedGuidedBackprop(GuidedBackprop):
    def __init__(self, 
                 model,
                 R_mask, 
                 **kwargs):
        super(_MaskedGuidedBackprop, self).__init__(model, neuron_selection_mode="all", **kwargs)
        self.initialize_r_mask(R_mask)

    def initialize_r_mask(self, R_mask):
        """0か1のベクトルでRが通れる道をターゲットのみに限定
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """
        初期化時に与えたOne-hotベクターとの要素積をとることでRが通れる道を限定
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R


class GBP(_MaskedGuidedBackprop):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 **kwargs):
        """ターゲット:predictionの値，それ以外0
        Arguments:
            model {[type]} -- [description]
            target_id {[type]} -- [description]
            predictions {[type]} -- [description]
        """
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        super(GBP, self).__init__(model, R_mask=R_mask, **kwargs)
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(GBP, self).analyze(inputs), 0)
        else:
            return super(GBP, self).analyze(inputs)
        
class _MaskedDeepTaylor(BoundedDeepTaylor):
    """Give any specific path to the DTD
    """

    def __init__(self, model, R_mask, **kwargs):
        super(_MaskedDeepTaylor, self).__init__(
            model, neuron_selection_mode="all", **kwargs)
        self.initialize_r_mask(R_mask)

    def initialize_r_mask(self, R_mask):
        """Mask R road
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """Multiplication with the initialized one-hot vector
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R


class LRP(_MaskedDeepTaylor):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 low=-1.,
                 high=1.,
                 **kwargs):
        """Target value:same as prediction，otherwise:0
        Arguments:
            model {[type]} -- [description]
            target_id {[type]} -- [description]
            predictions {[type]} -- [description]
        """
        self.relu = relu
        R_mask = np.zeros((model.output_shape[1]))
        R_mask[target_id] = 1
        super(LRP, self).__init__(model, R_mask=R_mask, low=low, high=high, **kwargs)
                
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(LRP, self).analyze(inputs), 0)
        else:
            return super(LRP, self).analyze(inputs)
        
        
class _LRPSubtraction(object):
    def __init__(self, 
                 model, 
                 target_id, 
                 scaling=True, 
                 **kwargs):
        self.model = model
        self.target_id = target_id
        self.scaling = scaling
        self.target_analyzer = self._get_target_analyzer(**kwargs)
        self.others_analyzer = self._get_others_analyzer(**kwargs)

    def _get_target_analyzer(self, **kwargs):
        raise NotImplementedError

    def _get_others_analyzer(self, **kwargs):
        raise NotImplementedError

    def analyze_target(self, inputs):
        return self.target_analyzer.analyze(inputs)

    def analyze_others(self, inputs):
        return self.others_analyzer.analyze(inputs)

    def analyze(self, inputs):
        analysis_target = self.analyze_target(inputs)
        analysis_others = self.analyze_others(inputs)

        # 画像毎にスケーリング調整
        equal_magnification = 1
        if self.scaling:
            equal_magnification = analysis_target.sum(
                axis=(1, 2, 3), keepdims=True) / analysis_others.sum(
                    axis=(1, 2, 3), keepdims=True)
        analysis = analysis_target - analysis_others * equal_magnification
        return analysis
    
    
class _CLRPBase(BoundedDeepTaylor):
    def __init__(self, 
                 model, 
                 target_id, 
                 **kwargs):
        super(_CLRPBase, self).__init__(
            model, neuron_selection_mode="all", **kwargs)
        self.target_id = target_id
        self.class_num = model.output_shape[1]
        self.initialize_r_mask()

    def initialize_r_mask(self):
        raise NotImplementedError

    def _head_mapping(self, X):
        target_value = Lambda(lambda x: (x[:, self.target_id]))(X)
        X = Lambda(lambda x: (x[:, None] * self.R_mask))(target_value)
        return X
    
class _CLRPTarget(_CLRPBase):
    def initialize_r_mask(self):
        R_mask = np.zeros(self.class_num)
        R_mask[self.target_id] = 1
        self.R_mask = K.constant(R_mask)


class _CLRPOthers(_CLRPBase):
    """R dual for CLRP1
    """

    def initialize_r_mask(self):
        R_mask = np.ones(self.class_num)
        R_mask[self.target_id] = 0
        R_mask /= self.class_num - 1
        self.R_mask = K.constant(R_mask)


class CLRP(_LRPSubtraction):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 low=-1.,
                 high=1.,
                 **kwargs):
        super(CLRP, self).__init__(model, target_id=target_id, low=low, high=high, **kwargs)
        self.relu = relu
        
    def _get_target_analyzer(self, **kwargs):
        return _CLRPTarget(self.model, target_id=self.target_id, **kwargs)

    def _get_others_analyzer(self, **kwargs):
        return _CLRPOthers(self.model, target_id=self.target_id, **kwargs)
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(CLRP, self).analyze(inputs), 0)
        else:
            return super(CLRP, self).analyze(inputs)
    
class _SGLRPBase(BoundedDeepTaylor):
    """Initialize R with Softmax Gradient
    """
    def __init__(self, 
                 model, 
                 target_id, 
                 **kwargs):
        super(_SGLRPBase, self).__init__(model, neuron_selection_mode="all", **kwargs)
        self.target_id = target_id
        self.class_num = model.output_shape[1]
        self.initialize_r_mask()

    def initialize_r_mask(self):
        raise NotImplementedError
    
   
    def _head_mapping(self, X):
        """
        target:yt(1-yt)  base: ytyi
        """
        # It's possible to have a perfect prediction. A small amount is added to prevent problems. 
        epsilon = 1e-8 
        
        # target is 1, else 0
        Kronecker_delta = np.zeros(self.class_num)
        Kronecker_delta[self.target_id] = 1
        Kronecker_delta = K.constant(Kronecker_delta)

        # target is 0, else 1
        Inv_Kronecker_delta = np.ones(self.class_num)
        Inv_Kronecker_delta[self.target_id] = 0
        Inv_Kronecker_delta = K.constant(Inv_Kronecker_delta)
        
        X = _SoftMax()(X)
        target_value = Lambda(lambda x: (x[:, self.target_id]))(X)


        X = Lambda(
            lambda x: (x * (1. - x) + epsilon) * Kronecker_delta + x * Inv_Kronecker_delta * target_value[:, None],
            output_shape=lambda input_shape: (None, int(input_shape[1])))(X)
        
        X = Lambda(lambda x: (self.R_mask * x))(X)
        return X


class _SGLRPTarget(_SGLRPBase):
    def initialize_r_mask(self):
        R_mask = np.zeros(self.class_num)
        R_mask[self.target_id] = 1
        self.R_mask = K.constant(R_mask)


class _SGLRPDual(_SGLRPBase):
    def initialize_r_mask(self):
        R_mask = np.ones(self.class_num)
        R_mask[self.target_id] = 0
        self.R_mask = K.constant(R_mask)


class SGLRP(_LRPSubtraction):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 low=-1.,
                 high=1.,
                 **kwargs):
        super(SGLRP, self).__init__(model, target_id=target_id, low=low, high=high, **kwargs)
        self.relu = relu
        
    def _get_target_analyzer(self, **kwargs):
        return _SGLRPTarget(self.model, target_id=self.target_id, **kwargs)

    def _get_others_analyzer(self, **kwargs):
        return _SGLRPDual(self.model, target_id=self.target_id, **kwargs)
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(SGLRP, self).analyze(inputs), 0)
        else:
            return super(SGLRP, self).analyze(inputs)


class _SGLRP2Base(BoundedDeepTaylor):
    """Initialize R with Softmax Gradient
    """
    def __init__(self, 
                 model, 
                 target_id, 
                 **kwargs):
        super(_SGLRP2Base, self).__init__(model, neuron_selection_mode="all", **kwargs)
        self.target_id = target_id
        self.class_num = model.output_shape[1]
        self.initialize_r_mask()

    def initialize_r_mask(self):
        raise NotImplementedError
    
    def _head_mapping(self, X):
        """
        target:yt，others:ytyj/(1-yt)
        """
        # It's possible to have a perfect prediction. A small amount is added to prevent problems. 
        epsilon = 1e-8 
        
        Kronecker_delta = np.zeros(self.class_num)
        Kronecker_delta[self.target_id] = 1
        Kronecker_delta = K.constant(Kronecker_delta)

        Inv_Kronecker_delta = np.ones(self.class_num)
        Inv_Kronecker_delta[self.target_id] = 0
        Inv_Kronecker_delta = K.constant(Inv_Kronecker_delta)

        target_value = Lambda(lambda x: (x[:, self.target_id]))(X)

        X = _SoftMax()(X)
        X = Lambda(
            lambda x: x * (Kronecker_delta) - x * (Inv_Kronecker_delta) * target_value[:, None] / (1 - target_value[:, None] + epsilon),
            output_shape=lambda input_shape: (None, int(input_shape[1])))(X)

        X = Lambda(lambda x: (self.R_mask * x))(X)
        return X


class _SGLRP2Target(_SGLRP2Base):
    def initialize_r_mask(self):
        R_mask = np.zeros(self.class_num)
        R_mask[self.target_id] = 1
        self.R_mask = K.constant(R_mask)


class _SGLRP2Dual(_SGLRP2Base):
    def initialize_r_mask(self):
        R_mask = np.ones(self.class_num)
        R_mask[self.target_id] = 0
        self.R_mask = K.constant(R_mask)
        
class SGLRP2(_LRPSubtraction):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 low=-1.,
                 high=1.,
                 **kwargs):
        super(SGLRP2, self).__init__(model, target_id=target_id, low=low, high=high, **kwargs)
        self.relu = relu
        
    def _get_target_analyzer(self, **kwargs):
        return _SGLRP2Target(self.model, target_id=self.target_id, **kwargs)

    def _get_others_analyzer(self, **kwargs):
        return _SGLRP2Dual(self.model, target_id=self.target_id, **kwargs)
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(SGLRP2, self).analyze(inputs), 0)
        else:
            return super(SGLRP2, self).analyze(inputs)
        
        
class GradCAM(object):
    def __init__(self,
                 model,
                 target_id,
                 layer_name="block5_pool",
                 relu=False,
                 **kwargs):

        class_output = model.output[:, target_id]

        conv_output = model.get_layer(
            layer_name).output  
        grads = K.gradients(class_output, conv_output)[
            0]  
        self.gradient_function = K.function(
            [model.input],
            [conv_output, grads],
        ) 
        self.relu = relu

    def analyze(self, inputs):
        outputs, grads_vals = self.gradient_function([inputs])

        weights = np.mean(grads_vals, axis=(1, 2))
        cams = (outputs * weights[:, np.newaxis, np.newaxis, :]).sum(
            axis=3, keepdims=True)
                
        resized_cams = resize(cams, np.shape(inputs), mode='reflect', anti_aliasing=True)

        if self.relu:
            return np.maximum(resized_cams, 0)
        else:
            return resized_cams
class _OAGLRPBase(BoundedDeepTaylor):
    """Initialize R with Output Activation Gradient
    """
    def __init__(self, 
                 model, 
                 target_id, 
                 **kwargs):
        super(_OAGLRPBase, self).__init__(model, neuron_selection_mode="all", **kwargs)
        self.target_id = target_id
        self.class_num = model.output_shape[1]
        self.initialize_r_mask()

    def initialize_r_mask(self):
        raise NotImplementedError
    
    def _head_mapping_oldmod(self, X):
        """
        target:yt，others:ytyj/(1-yt)
        """
        # It's possible to have a perfect prediction. A small amount is added to prevent problems. 
        epsilon = 1e-6 
        
        # target is 1, else 0
        Kronecker_delta = np.zeros(self.class_num)
        Kronecker_delta[self.target_id] = 1
        Kronecker_delta = K.constant(Kronecker_delta)

        # target is 0, else 1
        Inv_Kronecker_delta = np.ones(self.class_num)
        Inv_Kronecker_delta[self.target_id] = 0
        Inv_Kronecker_delta = K.constant(Inv_Kronecker_delta)

        target_value = Lambda(lambda x: (x[:, self.target_id]))(X)
        
        X = _Sigmoid()(X)
        X = Lambda(
            lambda x: x * (Kronecker_delta) - x * (Inv_Kronecker_delta) * target_value[:, None] / (1 - target_value[:, None] + epsilon),
            output_shape=lambda input_shape: (None, int(input_shape[1])))(X)
        
        # targetかdualかでマスクが変わる
        X = Lambda(lambda x: (self.R_mask * x))(X)
        return X
    
    def _head_mapping(self, X):
        """
        target:yt(1-yt)
        """
        
        # target is 1, else 0
        Kronecker_delta = np.zeros(self.class_num)
        Kronecker_delta[self.target_id] = 1
        Kronecker_delta = K.constant(Kronecker_delta)

        # target is 0, else 1
        Inv_Kronecker_delta = np.ones(self.class_num)
        Inv_Kronecker_delta[self.target_id] = 0
        Inv_Kronecker_delta = K.constant(Inv_Kronecker_delta)
        
        X = _Sigmoid()(X)
        target_value = Lambda(lambda x: (x[:, self.target_id]))(X)


        X = Lambda(
            lambda x: x * (1 - x) * Kronecker_delta + x * (1 - x) * Inv_Kronecker_delta, # should be plus
            output_shape=lambda input_shape: (None, int(input_shape[1])))(X)
        
        X = Lambda(lambda x: (self.R_mask * x))(X)
        return X
    
    def _head_mapping_sglrp(self, X):
        """
        target:yt(1-yt) 
        """
        # It's possible to have a perfect prediction. A small amount is added to prevent problems. 
        epsilon = 1e-6 
        
        # target is 1, else 0
        Kronecker_delta = np.zeros(self.class_num)
        Kronecker_delta[self.target_id] = 1
        Kronecker_delta = K.constant(Kronecker_delta)

        # target is 0, else 1
        Inv_Kronecker_delta = np.ones(self.class_num)
        Inv_Kronecker_delta[self.target_id] = 0
        Inv_Kronecker_delta = K.constant(Inv_Kronecker_delta)
        
        X = _Sigmoid()(X)
        target_value = Lambda(lambda x: (x[:, self.target_id]))(X)


        X = Lambda(
            lambda x: x * (1 - x) * Kronecker_delta + x * Inv_Kronecker_delta * target_value[:, None],
            output_shape=lambda input_shape: (None, int(input_shape[1])))(X)
        
        # targetかdualかでマスクが変わる
        X = Lambda(lambda x: (self.R_mask * x))(X)
        return X
    
    def _head_mapping_bad(self, X):
        """
        target:yt，others:ytyj/(1-yt)
        """
        # It's possible to have a perfect prediction. A small amount is added to prevent problems. 
        epsilon = 1e-6 
        
        # target is 1, else 0
        Kronecker_delta = np.zeros(self.class_num)
        Kronecker_delta[self.target_id] = 1
        Kronecker_delta = K.constant(Kronecker_delta)

        # target is 0, else 1
        Inv_Kronecker_delta = np.ones(self.class_num)
        Inv_Kronecker_delta[self.target_id] = 0
        Inv_Kronecker_delta = K.constant(Inv_Kronecker_delta)

        outputs = _Sigmoid()(X)
        target_value = Lambda(lambda x: (x[:, self.target_id]))(outputs)

        X = Lambda(
            lambda x: x * (target_value[:, None] * (1 - Kronecker_delta * outputs) - Inv_Kronecker_delta * outputs * target_value[:, None]),
            output_shape=lambda input_shape: (None, int(input_shape[1])))(X)
        
        # targetかdualかでマスクが変わる
        X = Lambda(lambda x: (self.R_mask * x))(X)
        return X


class _OAGLRPTarget(_OAGLRPBase):
    def initialize_r_mask(self):
        R_mask = np.zeros(self.class_num)
        R_mask[self.target_id] = 1
        self.R_mask = K.constant(R_mask)


class _OAGLRPDual(_OAGLRPBase):
    def initialize_r_mask(self):
        R_mask = np.ones(self.class_num)
        R_mask[self.target_id] = 0
        self.R_mask = K.constant(R_mask)


class OAGLRP(_LRPSubtraction):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 low=-1.,
                 high=1.,
                 **kwargs):
        super(OAGLRP, self).__init__(model, target_id=target_id, low=low, high=high, **kwargs)
        self.relu = relu
        
    def _get_target_analyzer(self, **kwargs):
        return _OAGLRPTarget(self.model, target_id=self.target_id, **kwargs)

    def _get_others_analyzer(self, **kwargs):
        return _OAGLRPDual(self.model, target_id=self.target_id, **kwargs)
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(OAGLRP, self).analyze(inputs), 0)
        else:
            return super(OAGLRP, self).analyze(inputs)
        
        
class GuidedGradCAM(object):
    def __init__(self, 
                 model, 
                 target_id, 
                 layer_name="block5_pool",
                 relu=False,
                 **kwargs):
        self.model = model
        self.target_id = target_id
        self.relu = relu
        self.gradcam = GradCAM(self.model, target_id=self.target_id, layer_name=layer_name, relu=relu, **kwargs)
        self.guidedbackprop = GBP(self.model, target_id=self.target_id, relu=relu, **kwargs)
                
    def analyze(self, inputs):
        return self.gradcam.analyze(inputs) * self.guidedbackprop.analyze(inputs)