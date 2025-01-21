import torch
import torch.nn as nn

def quantize_tensor(tensor, scale, zero_point):
    # print(tensor)
    q_tensor = (tensor / scale).round() + zero_point
    
    return q_tensor.clamp(0, 255).to(torch.uint8)

def dequantize_tensor(q_tensor, scale, zero_point):
    
    return (q_tensor.to(torch.float32) - zero_point) * scale

def dequantize_static_tensor(q_tensor):
    return q_tensor

def calculate_dynamic_scale_and_zero_point(tensor):
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / 255.0
    zero_point = (-min_val / scale).round()
    return scale, zero_point

def quantize_model(model):
    quantized_weights = {}
    scales = {}
    zero_points = {}

    for name, param in model.named_parameters():
     
        if 'weight' in name:
            
            scale, zero_point = calculate_dynamic_scale_and_zero_point(param.data)
            quantized_weights[name] = quantize_tensor(param.data, scale, zero_point)
            
            scales[name] = scale
            zero_points[name] = zero_point
        else:
            quantized_weights[name] = param.data
          
    
    return quantized_weights, scales, zero_points

def apply_quantized_weights(model, quantized_weights, scales, zero_points):
    for name, param in model.named_parameters():
        if 'weight' in name:
            scale = scales[name]
            zero_point = zero_points[name]
            param.data = dequantize_tensor(quantized_weights[name], scale, zero_point)
        else:
            param.data = quantized_weights[name]
    return model


def load_quantized_model(model):
    checkpoint = torch.load(f'quantized_model_lenet.pth')

    model_quantized_weights = checkpoint['model_state_dict']
    model_scales = checkpoint['model_scales']
    model_zero_points = checkpoint['model_zero_points']

    
    model = apply_quantized_weights(model, model_quantized_weights, model_scales, model_zero_points)
    
    return model