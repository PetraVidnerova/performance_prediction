import sys
import pandas as pd
import torch

sys.path.append("HW-NAS-Bench")

from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
from thop import profile




hw_api = HWAPI("HW-NAS-Bench/HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")



df_list = []
input_ = torch.randn(1, 3, 32, 32)

for i in range(15625):
    if i % 100 == 0:
        print("**************************************")
        print(i)
        print("**************************************")
        
    config = hw_api.get_net_config(i, "cifar10")
    network = get_cell_based_tiny_net(config)
    
    macs, params = profile(network, inputs=(input_,))
    df_list.append(
        {
            "net": config["arch_str"],
            "macs": macs
        }
    )



df = pd.DataFrame(df_list).set_index("net")
print(df)



df.to_csv("mydata/network_macs.csv")





