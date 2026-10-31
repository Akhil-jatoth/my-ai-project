import torch
import torch.nn as nn
from model.crowd_model import MCNN

def inspect_model():
    model = MCNN()
    # Set final layer to Linear Conv2d(30, 1) as proven in Phase 7
    model.density_layer = nn.Conv2d(30, 1, kernel_size=1)
    
    print("=" * 80)
    print("PHASE 9: MCNN ARCHITECTURAL VERIFICATION & PARAMETER BREAKDOWN")
    print("=" * 80)
    
    total_params = 0
    for name, module in model.named_children():
        print(f"\n[{name.upper()}]")
        if isinstance(module, nn.Sequential):
            for i, layer in enumerate(module):
                l_params = sum(p.numel() for p in layer.parameters())
                total_params += l_params
                if isinstance(layer, nn.Conv2d):
                    print(f"  ({i}) Conv2d: in={layer.in_channels}, out={layer.out_channels}, kernel={layer.kernel_size}, stride={layer.stride}, padding={layer.padding} | Params: {l_params:,}")
                elif isinstance(layer, nn.MaxPool2d):
                    print(f"  ({i}) MaxPool2d: kernel={layer.kernel_size}, stride={layer.stride}, padding={layer.padding} | Params: 0")
                elif isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                    print(f"  ({i}) {layer.__class__.__name__} | Params: 0")
                else:
                    print(f"  ({i}) {layer} | Params: {l_params:,}")
        else:
            l_params = sum(p.numel() for p in module.parameters())
            total_params += l_params
            if isinstance(module, nn.Conv2d):
                print(f"  Conv2d (1x1 Fusion): in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size}, stride={module.stride}, padding={module.padding} | Params: {l_params:,}")
            else:
                print(f"  {module} | Params: {l_params:,}")
                
    print("\n" + "-" * 80)
    print(f"TOTAL TRAINABLE PARAMETERS: {total_params:,}")
    print("=" * 80)

if __name__ == '__main__':
    inspect_model()

