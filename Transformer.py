import torch
from allennlp.nn.util import get_final_encoder_states
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from overrides import overrides

@Seq2VecEncoder.register("seq2vec_transformer_encoder")
class TransformerSeq2VecEncoder(Seq2VecEncoder):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int) -> None:
    
        super(TransformerSeq2VecEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.stacked_self_att_enc = StackedSelfAttentionEncoder(input_dim=input_dim,
                                                                hidden_dim=hidden_dim,
                                                                projection_dim=projection_dim,
                                                                feedforward_hidden_dim=feedforward_hidden_dim,
                                                                num_layers=num_layers,
                                                                num_attention_heads=num_attention_heads)
        
    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        out = self.stacked_self_att_enc(inputs,mask)
        return get_final_encoder_states(out, mask)
            
    