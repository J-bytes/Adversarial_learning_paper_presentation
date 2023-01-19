import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
import matplotlib.pyplot as plt

names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def main() :
    make_adv = True
    ds = CIFAR('/mnt/f/cifar-100-python/')
    model, _ = make_and_restore_model(arch='resnet50', dataset=ds)
    model.eval()
    attack_kwargs = {
       'constraint': 'inf', # L-inf PGD
       'eps': 0.05, # Epsilon constraint (L-inf norm)
       'step_size': 0.01, # Learning rate for PGD
       'iterations': 100, # Number of PGD steps
       'targeted': True, # Targeted attack
       'custom_loss': None # Use default cross-entropy loss
    }

    _, test_loader = ds.make_loaders(workers=4, batch_size=16)
    im, label = next(iter(test_loader))
    target_label = (label + ch.randint_like(label, high=9)) % 10
    adv_out, adv_im = model(im, target_label, make_adv, **attack_kwargs)
    print(adv_out.shape, adv_im.shape)
    #plt.imshow(adv_out[0].detach().cpu().numpy())
    #plt.show()
    for im in adv_im :
        plt.imshow(im.detach().cpu().permute(1,2,0).numpy())
        plt.show()

if __name__ == '__main__':
    main()


