from torch.autograd import Variable
import torch.onnx
import torchvision
import argparse
import torch 


parser = argparse.ArgumentParser(description="Convert pth2onnx")
parser.add_argument("--input", type=str, default="best_model.pth")
parser.add_argument("--output", type=str, default="convert.onnx")
args = parser.parse_args()



def convert2onnx(): 

    cuda = torch.device("cuda")
    model = torch.load(args.input, map_location=cuda)
    model.eval() 
    dummy_input = Variable(torch.randn(1, 3, 640, 640))

    torch.onnx.export(
         model,          
         dummy_input.to(cuda),        
         args.output,opset_version=11,         
         export_params=True) 
    
    print(" ") 
    print('Model has been converted to ONNX') 


if __name__ == '__main__':
    convert2onnx()
