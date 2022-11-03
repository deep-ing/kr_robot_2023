import os 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.rc('font', weight='bold') 
from tensorflow.python.summary.summary_iterator import summary_iterator

def convert_tfevent(filepath):
    return pd.DataFrame([
        parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
    ])

def parse_tfevent(tfevent):
    return dict(
        wall_time=tfevent.wall_time,
        name=tfevent.summary.value[0].tag,
        step=tfevent.step,
        value=float(tfevent.summary.value[0].simple_value),
    )

if __name__ == "__main__":    
    runs=None 
    alls=[] 
    for i, run in enumerate(runs):
            a = convert_tfevent(run)
            a['seed'] = i
            a['model'] = {"anneal_dist":"dist", "anneal_pred":"pred", "baseline": "baseline"}[model]
            alls.append(a[a['name']=='CReQuaL/entropy_1'])
            alls.append(a[a['name']=='CReQuaL/entropy_0'])
    df = pd.concat(alls)
    df['code'] = df['name'].apply(lambda x : x[-1])
