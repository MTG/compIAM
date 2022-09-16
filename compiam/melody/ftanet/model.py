from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Lambda, \
    GlobalAveragePooling2D, MaxPooling2D, Concatenate, Add, Multiply, \
        Softmax, Reshape, UpSampling2D, Conv1D


class FTANet(object):
    """FTA-Net predominant melody extraction
    """
    def __init__(self):
        """FTA-Net predominant melody extraction init method
        """
        self.ftanet = self.create_ftanet()

    @staticmethod
    def SF_Module(x_list, n_channel, reduction, limitation):
        """TODO
        Args:
            TODO
        Returns:
            TODO
        """
        ## Split
        fused = None
        for x_s in x_list:
            if fused==None:
                fused = x_s
            else:
                fused = Add()([fused, x_s])
            
        ## Fuse
        fused = GlobalAveragePooling2D()(fused)
        fused = BatchNormalization()(fused)
        fused = Dense(max(n_channel // reduction, limitation), activation='selu')(fused)

        ## Select
        masks = []
        for i in range(len(x_list)):
            masks.append(Dense(n_channel)(fused))
        mask_stack = Lambda(K.stack, arguments={'axis': -1})(masks)
        # (n_channel, n_kernel)
        mask_stack = Softmax(axis=-2)(mask_stack)

        selected = None
        for i, x_s in enumerate(x_list):
            mask = Lambda(lambda z: z[:, :, i])(mask_stack)
            mask = Reshape((1, 1, n_channel))(mask)
            x_s = Multiply()([x_s, mask])
            if selected==None:
                selected = x_s
            else:
                selected = Add()([selected, x_s])
        return selected

    @staticmethod
    def FTA_Module(x, shape, kt, kf):
        """TODO
        Args:
            TODO
        Returns:
            TODO
        """
        x = BatchNormalization()(x)

        ## Residual
        x_r = Conv2D(shape[2], (1, 1), padding='same', activation='relu')(x)

        ## Time Attention
        # Attn Map (1, T, C), FC
        a_t = Lambda(K.mean, arguments={'axis': -3})(x)
        a_t = Conv1D(shape[2], kt, padding='same', activation='selu')(a_t)
        a_t = Conv1D(shape[2], kt, padding='same', activation='selu')(a_t) #2
        a_t = Softmax(axis=-2)(a_t)
        a_t = Reshape((1, shape[1], shape[2]))(a_t)
        # Reweight
        x_t = Conv2D(shape[2], (3, 3), padding='same', activation='selu')(x)
        x_t = Conv2D(shape[2], (5, 5), padding='same', activation='selu')(x_t)
        x_t = Multiply()([x_t, a_t])

        # Frequency Attention
        # Attn Map (F, 1, C), Conv1D
        a_f = Lambda(K.mean, arguments={'axis': -2})(x)
        a_f = Conv1D(shape[2], kf, padding='same', activation='selu')(a_f)
        a_f = Conv1D(shape[2], kf, padding='same', activation='selu')(a_f)
        a_f = Softmax(axis=-2)(a_f)
        a_f = Reshape((shape[0], 1, shape[2]))(a_f)
        # Reweight
        x_f = Conv2D(shape[2], (3, 3), padding='same', activation='selu')(x)
        x_f = Conv2D(shape[2], (5, 5), padding='same', activation='selu')(x_f)
        x_f = Multiply()([x_f, a_f])

        return x_r, x_t, x_f

    def create_ftanet(self, input_shape=(320, 128, 3)):
        """TODO
        Args:
            TODO
        Returns:
            TODO
        """
        visible = Input(shape=input_shape)
        x = BatchNormalization()(visible)

        ## Bottom
        # bm = BatchNormalization()(x)
        bm = x
        bm = Conv2D(16, (4, 1), padding='valid', strides=(4, 1), activation='selu')(bm) # 80
        bm = Conv2D(16, (4, 1), padding='valid', strides=(4, 1), activation='selu')(bm) # 20
        bm = Conv2D(16, (4, 1), padding='valid', strides=(4, 1), activation='selu')(bm) # 5
        bm = Conv2D(1,  (5, 1), padding='valid', strides=(5, 1), activation='selu')(bm) # 1

        shape=input_shape
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0], shape[1], 32), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 32, 4, 4)
        x = MaxPooling2D((2, 2))(x)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0]//2, shape[1]//2, 64), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 64, 4, 4)
        x = MaxPooling2D((2, 2))(x)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0]//4, shape[1]//4, 128), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 128, 4, 4)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0]//4, shape[1]//4, 128), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 128, 4, 4)

        x = UpSampling2D((2, 2))(x)
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0]//2, shape[1]//2, 64), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 64, 4, 4)

        x = UpSampling2D((2, 2))(x)
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0], shape[1], 32), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 32, 4, 4)
        
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0], shape[1], 1), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 1, 4, 4)
        x = Concatenate(axis=1)([bm, x])
    
        # Softmax
        x = Lambda(K.squeeze, arguments={'axis': -1})(x)
        x = Softmax(axis=-2)(x)
        return Model(inputs=visible, outputs=x)
