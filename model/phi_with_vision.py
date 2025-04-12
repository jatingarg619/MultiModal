import torch
import torch.nn as nn

class PhiWithVision(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.phi = base_model
        
        # Add a projection layer for image embeddings
        self.image_projection = nn.Linear(
            in_features=512,  # SigLIP embedding dimension
            out_features=self.phi.config.hidden_size  # Phi model dimension
        )
        
        # Copy necessary attributes from base model
        self.config = base_model.config
        self.prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
        self.generation_config = base_model.generation_config
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, input_ids=None, attention_mask=None, image_embeddings=None, 
                labels=None, inputs_embeds=None, **kwargs):
        # If inputs_embeds is provided, use it directly
        if inputs_embeds is not None:
            combined_embeddings = inputs_embeds
        else:
            # Debug prints
            print(f"Image embeddings shape: {image_embeddings.shape}")
            
            # Project image embeddings to match model dimensions
            if image_embeddings.dim() == 4:
                print("Reshaping 4D image embeddings...")
                image_embeddings = image_embeddings.squeeze()  # Remove extra dimensions
                print(f"After squeeze shape: {image_embeddings.shape}")
            
            projected_image = self.image_projection(image_embeddings)
            print(f"Projected image shape: {projected_image.shape}")
            
            # Get text embeddings from first layer
            text_embeddings = self.phi.get_input_embeddings()(input_ids)
            print(f"Text embeddings shape: {text_embeddings.shape}")
            
            # Make projected_image 3D to match text_embeddings
            projected_image = projected_image.unsqueeze(1)
            print(f"Projected image after unsqueeze shape: {projected_image.shape}")
            
            # Now both tensors should be 3D
            combined_embeddings = torch.cat([projected_image, text_embeddings], dim=1)
            print(f"Combined embeddings shape: {combined_embeddings.shape}")
            
            # Adjust attention mask
            if attention_mask is not None:
                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device
                )
                attention_mask = torch.cat([extended_attention_mask, attention_mask], dim=1)
        
        # Forward pass through the model
        outputs = self.phi(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs

    def get_input_embeddings(self):
        return self.phi.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.phi.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.phi.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.phi.set_output_embeddings(new_embeddings) 