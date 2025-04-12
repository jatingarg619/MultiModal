import torch
import torch.nn as nn

class PhiWithVision(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.phi = base_model
        
        # Add a projection layer for image embeddings
        self.image_projection = nn.Linear(
            in_features=base_model.config.hidden_size,  # SigLIP embedding dimension
            out_features=base_model.config.hidden_size  # Phi model dimension
        )
        
    def forward(self, input_ids, attention_mask, image_embeddings, labels=None):
        # Project image embeddings to match model dimensions
        projected_image = self.image_projection(image_embeddings)
        
        # Get text embeddings from first layer
        text_embeddings = self.phi.get_input_embeddings()(input_ids)
        
        # Concatenate image and text embeddings along sequence length dimension
        combined_embeddings = torch.cat([projected_image.unsqueeze(1), text_embeddings], dim=1)
        
        # Adjust attention mask to account for added image token
        extended_attention_mask = torch.ones(
            (attention_mask.shape[0], 1),
            device=attention_mask.device
        )
        attention_mask = torch.cat([extended_attention_mask, attention_mask], dim=1)
        
        # Forward pass through the model
        outputs = self.phi(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs 