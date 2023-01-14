
from model import create_vgg_model
from data_setup import preprocess, deprocess, get_features, gram_matrix
from loss_functions import content_loss, style_loss, total_loss
from torch import optim
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image


def predict(content_image, style_image):

    # Create model
    model = create_vgg_model() 

    # Transform images
    content_img = preprocess(content_image) 
    style_img = preprocess(style_image)
    target_img = content_img.clone().requires_grad_(True)

    content_features = get_features(content_img, model) 
    style_features = get_features(style_img, model)
    
    style_gram = {layer: gram_matrix(style_features[layer]) for layer in style_features}

   # Inference
    optimizer = optim.Adam([target_img], lr=0.06)

    alpha_param = 1
    beta_param = 1e2
    epochs = 60
  
    for i in range(epochs):
      target_features = get_features(target_img, model)

      c_loss = content_loss(target_features['layer_4'], content_features['layer_4'])
      s_loss = style_loss(target_features, style_gram)
      t_loss = total_loss(c_loss, s_loss, alpha_param, beta_param)

      optimizer.zero_grad()
      t_loss.backward()
      optimizer.step()
        
    results = deprocess(target_img)

    return Image.fromarray((results * 255).astype(np.uint8))

# Gradio Interface

example_list =  [['content/content1.jpg',
                  'style/style1.jpg'],
                ['content/content2.jpg',
                  'style/style2.jpg'],
                ['content/content3.jpg',
                  'style/style3.jpg']]

title = "Neural Style Transfer 🎨"
description = "It will take about 1 minute for the result to be displayed. Since the algorithm runs a small number of epochs (to reduce the waiting time), the result will not always be as good as it should be."

demo = gr.Interface(fn=predict,
                    inputs=['image', 'image'],
                    outputs=gr.Image().style(width=256, height=256),
                    examples=example_list,
                    title=title,
                    description=description)


demo.launch(debug=False,
            share=False)
