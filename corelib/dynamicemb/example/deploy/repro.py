
import torch
import ops
import dynamicemb

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # This is the problematic line from model.py
        x = torch.ops.dynamicemb.ir_emb_lookup(x, x.shape[0], [16])
        return x

def test():
    device = "cpu"
    model = Model().to(device=device)
    x = torch.randn(8, 10, device=device)
    
    print("Attempting to run model...")
    try:
        out = model(x)
        print("Model ran successfully")
        print("Output type:", type(out))
    except Exception as e:
        print("Model failed as expected:")
        print(e)

if __name__ == "__main__":
    test()


