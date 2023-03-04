import torch
import torch.nn.functional as F
import numpy as np
# Crear un tensor de ejemplo codificado con label encoder
y = torch.randint(0, 2, (3, 2, 2))

x = torch.randn((3,2,2,2))
x = F.sigmoid(x) 
# Calcular la cantidad de categorías únicas en el tensor
num_classes = y.max() + 1

# Convertir el tensor codificado con label encoder a one hot encoder
tensor_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes)
y = tensor_one_hot.permute(0, 3, 1,2 )

"""
Multiplicacion
"""

interseccion = torch.mul(x, y)*2. # multiplicacion componente por componente
# Suma cada matriz de cada canal y me quedan las dos clases y los baches
suma_por_canal = interseccion.sum(dim=(2,3))

# suma los canales por lote
ans1 = suma_por_canal.sum(dim=0)
"""
Suma de cardinales por clase
"""

x_por_canal = x.sum(dim=(2,3))
y_por_canal = y.sum(dim=(2,3))

union = x_por_canal+y_por_canal
ans2 = union.sum(dim=0)

print(ans1/ans2)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim())) # para que este orden?
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1) #memoria contigua y operaciones lineales



def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input ).sum(-1) + (target ).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

print(compute_per_channel_dice(x,y))
#https://www.jeremyjordan.me/semantic-segmentation/
#https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py