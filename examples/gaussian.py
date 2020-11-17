from poisoning import xiao2018, frederickson2018
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import json

def draw_plot(Data, iAttacks, fAttacks, Projection, **kwargs):
    
    Data = (np.array(Data[0]), np.array(Data[1]))
    iAttacks = np.array(iAttacks)
    fAttacks = np.array(fAttacks)
    
    filename = kwargs.pop('filename', 'plot.png')
    aLabels = kwargs.pop('aLabels', [])
    aHandles = kwargs.pop('aHandles', [])
    title = kwargs.pop('title', '')
    center = kwargs.pop('center', (0,0))
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    
    ax.scatter(*Data[0].T, c='blue', label='Dataset Label 1')
    ax.scatter(*Data[1].T, c='red', label='Dataset Label -1')
    ax.scatter(*iAttacks.T, c='green', marker="D", label='Start attack points.')
    ax.scatter(*fAttacks.T, c='orange', marker="X", label='Optimized attack points.')

    if Projection is not None:
        # if type(Projection) not in [tuple, list]:
        #     ax.add_patch(Rectangle((center[0] - Projection, center[1] - Projection), Projection*2, Projection*2, edgecolor='black', fill=False))
        # else:
        #     ax.add_patch(Rectangle((center[0] - sum(Projection)/len(Projection), center[1] - sum(Projection)/len(Projection)), sum(Projection)*2/len(Projection), sum(Projection)*2/len(Projection), edgecolor='black', fill=False))
        if type(Projection) not in [tuple, list]:
            ax.add_patch(Rectangle((center[0] - Projection, center[1] - Projection), Projection*2, Projection*2, edgecolor='black', fill=False))
        else:
            ax.add_patch(Rectangle((Projection[0], Projection[0]), Projection[1]-Projection[0], Projection[1]-Projection[0], edgecolor='black', fill=False))


    handles, labels = ax.get_legend_handles_labels()
    handles += aHandles
    labels += aLabels
    
    legend = ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()

def write_comparative_rand(filename, size, n_attacks, labeler, randrange, projection, **kwargs):
    
    different = kwargs.pop('different', False)
    dist = kwargs.pop('distribution', False)
    
    if not dist:
        if different:
            X = [[random.randrange(*randrange[0]), random.randrange(*randrange[1])] for i in range(size)]
        else:
            X = [[random.randrange(*randrange), random.randrange(*randrange)] for i in range(size)]
    else:
        if different:
            X = [[np.random.normal(*randrange), np.random.normal(*randrange)] for i in range(size)]
        else:
            X = np.random.normal(*randrange, size=(size, 2)).tolist()

    Y = labeler(X)

    Attacks = random.sample(X, n_attacks)
    Labels = labeler(Attacks)

    with open(filename, 'w') as outfile:
            json.dump([X, Y, Attacks, Labels, projection], outfile)

def comparative_plots(json_file, **kwargs):
    
    def separate_by_label(X, Y):
        a = []
        b = []
        for i, item in enumerate(X):
            if Y[i] == 1:
                a.append(item)
            elif Y[i] == -1:
                b.append(item)
                
        return np.array(a), np.array(b)
    
    arguments = kwargs.pop('args', [{}])
    folderpath = kwargs.pop('path', 'compare')
    types = kwargs.pop('types', ['l1', 'l2', 'elastic'])
    center = kwargs.pop('center', None)
    algor = kwargs.pop('algorithm', xiao2018)

    with open(json_file, 'r') as infile:
        data = json.load(infile)

    X = data[0]
    Y = data[1]
    Attacks = data[2]
    Labels = data[3]
    projection = data[4]
    
    mod_proj = (projection[0], projection[1]) if type(projection) in [list, tuple] else projection

    for i, arg in enumerate(arguments):
        for tp in types:
            model = algor(type=tp, **arg)
            
            res = model.run(X, Y, Attacks, Labels, mod_proj)
            # print(res)
            fmt_kwargs = [f'{first}: {second}' for first, second in zip(arg.keys(), [str(a) for a in arg.values()])]
            draw_plot(separate_by_label(X, Y), Attacks, res, projection, center=center,
                        filename=f'{folderpath}/compare_{json_file.rstrip(".json")}_{model.algorithm_type}_{i}.png',
                        aHandles=[Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * (len(arg) + 7),
                        aLabels=['', ' - Kwargs -', '\n'.join(fmt_kwargs), '', f'Dataset Size: {len(X)}', f'Lambda: {model.alpha}', f'Attacks: {len(Attacks)}', f'Projection: {projection}'],
                        title=f'Compare Scatter ({model.algorithm_type}) ({json_file}) :: #{i+1}')

if __name__ == "__main__":

    # write_comparative_rand('inp.json', 200, 20, lambda X: [1 if x[0] < 10 else -1 for x in X], (10, 10), (-40, 60), distribution=True)
    # comparative_plots('inp.json', path='compare/xiao', algorithm=xiao2018)
    comparative_plots('inp.json', path='compare/fred', algorithm=frederickson2018)