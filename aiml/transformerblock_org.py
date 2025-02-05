import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
        GlobalAveragePooling1D,
        Layer,
        MultiHeadAttention,
        Dense,
        LayerNormalization,
        Dropout,
)
import matplotlib.pyplot as plt


def step_decay(epoch):
        """
        Step decay function for learning rate scheduling.

        This function gradually decreases the learning rate over epochs.

        Args:
                epoch (int): Current epoch number.

        Returns:
                float: New learning rate.
        """
        initial_lr = 0.001  # Initial learning rate
        drop = 0.5  # Factor by which the learning rate is dropped
        epochs_drop = 20.0  # Number of epochs after which the learning rate is dropped
        lr = initial_lr * (
                drop ** np.floor((1 + epoch) / epochs_drop)
        )  # Calculate the new learning rate
        return lr


class TransformerBlock(tf.keras.layers.Layer):
        """
        Represents a block in the Transformer model.

        A Transformer block consists of a multi-head attention layer, a feedforward network,
        layer normalization, and dropout.

        Args:
                d_model (int): Dimensionality of the embedding.
                num_heads (int): Number of attention heads.
                dff (int): Dimensionality of the feedforward network.
                rate (float, optional): Dropout rate. Defaults to 0.1.
                l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.01.

        Attributes:
                mha (MultiHeadAttention): Multi-head attention layer.
                ffn (Sequential): Feedforward network.
                layernorm1 (LayerNormalization): First layer normalization layer.
                layernorm2 (LayerNormalization): Second layer normalization layer.
                dropout1 (Dropout): First dropout layer.
                dropout2 (Dropout): Second dropout layer.
                gap (GlobalAveragePooling1D): Global average pooling layer.
        """
        def __init__(
                self, d_model, num_heads, dff, rate=0.1, l2_reg=0.01, **kwargs
        ):
                """
                Initializes a TransformerBlock instance.

                Args:
                        d_model (int): Dimensionality of the embedding.
                        num_heads (int): Number of attention heads.
                        dff (int): Dimensionality of the feedforward network.
                        rate (float, optional): Dropout rate. Defaults to 0.1.
                        l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.01.
                        **kwargs: Additional keyword arguments for the Layer class.
                """
                super().__init__(**kwargs)
                self.d_model = d_model
                self.num_heads = num_heads
                self.dff = dff
                self.rate = rate
                self.l2_reg = l2_reg

                self.mha = MultiHeadAttention(
                        key_dim=d_model, num_heads=num_heads
                )  # Multi-head attention layer

                # Apply L2 regularization to Dense layers
                self.ffn = Sequential(
                        [
                                Dense(
                                        dff, activation="relu", kernel_regularizer=l2(l2_reg)
                                ),  # First dense layer with ReLU activation
                                Dense(
                                        d_model, kernel_regularizer=l2(l2_reg)
                                ),  # Second dense layer
                        ]
                )  # Feedforward network
                self.layernorm1 = LayerNormalization(
                        epsilon=1e-6
                )  # First layer normalization layer
                self.layernorm2 = LayerNormalization(
                        epsilon=1e-6
                )  # Second layer normalization layer
                self.dropout1 = Dropout(rate)  # First dropout layer
                self.dropout2 = Dropout(rate)  # Second dropout layer
                self.gap = GlobalAveragePooling1D()  # Global average pooling layer

        def call(self, x, training=False):
                """
                Executes the Transformer block.

                This method performs the forward pass of the Transformer block, applying multi-head attention,
                layer normalization, dropout, and the feedforward network.

                Args:
                        x (Tensor): Input tensor.
                        training (bool, optional): Whether in training mode. Defaults to False.

                Returns:
                        Tensor: Output tensor.
                """

                attn_output = self.mha(
                        x, x, x
                )  # Apply multi-head attention to the input tensor
                attn_output = self.dropout1(
                        attn_output, training=training
                )  # Apply dropout to the attention output
                attn_output = self.layernorm1(
                        x + attn_output
                )  # Add the input tensor and normalize the result

                ffn_output = self.ffn(
                        attn_output
                )  # Apply the feedforward network to the normalized attention output
                ffn_output = self.dropout2(
                        ffn_output, training=training
                )  # Apply dropout to the feedforward output
                return self.layernorm2(
                        attn_output + ffn_output
                )  # Add the normalized attention output and normalize the result

        def get_config(self):
                """
                Gets the configuration of the Transformer block.

                Returns:
                        dict: Configuration dictionary for the Transformer block.
                """
                config = super(TransformerBlock, self).get_config()
                config.update(
                        {
                                "d_model": self.d_model,
                                "num_heads": self.num_heads,
                                "dff": self.dff,
                                "rate": self.rate,
                                "l2_reg": self.l2_reg,
                        }
                )
                return config

        @classmethod
        def from_config(cls, config):
                """
                Creates a Transformer block from its configuration.

                Args:
                        config (dict): Configuration dictionary for the Transformer block.

                Returns:
                        TransformerBlock: Transformer block created from the configuration.
                """
                return cls(**config)
