from mf_npe.flows.train_flows import XEncoder
from sbi.neural_nets.embedding_nets import CNNEmbedding
import numpy as np
from torch import nn

def _generate_embedding_networks(self, x_lf=None, x_hf=None):
    def build(kind: str, in_dim: int, which: str):
        if kind == "xEncoder":
            return XEncoder(
                input_dim=in_dim,
                hidden_dim=self.config_model["n_hidden_features"],
                output_dim=self.x_dim_out,
            ).to(self.device)
        if kind == "identity":
            return nn.Identity()
        
        if kind == "cnn":
            x_dim_hf_unflattened = int(np.sqrt(self.x_dim_hf))
            
            # For gaussian blob experiment
            embedding_net = CNNEmbedding(
                input_shape=(x_dim_hf_unflattened, x_dim_hf_unflattened), # 1024, you can also put a tuple such as 32x32 for instance
                in_channels=1,
                out_channels_per_layer=[6],
                num_conv_layers=1,
                num_linear_layers=1,
                output_dim=self.x_dim_out,
                kernel_size=5,
                pool_kernel_size=8
            )
            
            return embedding_net
        
        raise ValueError(
            f"Unknown type_embedding_{which}: {kind}. Choose 'xEncoder' or 'identity'."
        )

    if x_hf is not None:
        if self.x_dim_hf != x_hf.shape[1]:
            raise ValueError(f"HF data has {x_hf.shape[1]} features, but {self.x_dim_hf} were expected. Try to regenerate the data (by adding --generate_true_data and --generate_train_data to the train.py command) or change the x_dim_hf parameter.")

    embedding_lf = build(self.type_embedding_lf, self.x_dim_hf, "lf")
    embedding_hf = build(self.type_embedding_hf, self.x_dim_hf, "hf")

    return embedding_lf, embedding_hf