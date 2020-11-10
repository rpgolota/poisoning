from poisoning import xiao2018
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import json

def separate_by_label(X, Y):
    a = []
    b = []
    for i, item in enumerate(X):
        if Y[i] == 1:
            a.append(item)
        elif Y[i] == -1:
            b.append(item)
            
    return np.array(a), np.array(b)

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
    
    arguments = kwargs.pop('args', [{}])
    folderpath = kwargs.pop('path', 'compare')
    types = kwargs.pop('types', ['l1', 'l2', 'elastic'])
    center = kwargs.pop('center', None)

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
            model = xiao2018(type=tp, **arg)
            
            res = model.run(X, Y, Attacks, Labels, mod_proj)
            # print(res)
            fmt_kwargs = [f'{first}: {second}' for first, second in zip(arg.keys(), [str(a) for a in arg.values()])]
            draw_plot(separate_by_label(X, Y), Attacks, res, projection, center=center,
                        filename=f'{folderpath}/compare_{json_file.rstrip(".json")}_{model.algorithm_type}_{i}.png',
                        aHandles=[Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * (len(arg) + 7),
                        aLabels=['', ' - Kwargs -', '\n'.join(fmt_kwargs), '', f'Dataset Size: {len(X)}', f'Lambda: {model.alpha}', f'Attacks: {len(Attacks)}', f'Projection: {projection}'],
                        title=f'Compare Scatter ({model.algorithm_type}) ({json_file}) :: #{i+1}')

def gaussian_plots(**kwargs):
    
    number = kwargs.pop('runs', 10)
    n_attack = kwargs.pop('num_attack', 1)
    data_dist = kwargs.pop('distribution', (0, 1))
    label = kwargs.pop('labeler', lambda X: [-1 for i in X])
    size = kwargs.pop('samples', 20)
    types = kwargs.pop('types', ['l1', 'l2', 'elastic'])
    projection = kwargs.pop('projection', 5)
    modelkwargs = kwargs.pop('args', {})
    folderpath = kwargs.pop('path', 'images')
    
    if type(modelkwargs) is not list:
        modelkwargs = [modelkwargs]
    
    j = 0
    for arguments in modelkwargs:
        for tp in types:
            for i in range(number):
                model = xiao2018(type=tp, **arguments)
                X = np.random.normal(*data_dist, size=(size, 2))
                Y = label(X)
                
                combined = np.hstack((X, np.array([Y]).T))
                Attacks = random.sample(list(combined), n_attack)
                Labels = [row[-1] for row in Attacks]
                Attacks = [row[:-1] for row in Attacks]
                
                res = model.run(X, Y, Attacks, Labels, projection)
                
                fmt_kwargs = [f'{first}: {second}' for first, second in zip(arguments.keys(), [str(a) for a in arguments.values()])]
                draw_plot(separate_by_label(X, Y), Attacks, res, projection, center=(data_dist[0], data_dist[0]),
                          filename=f'{folderpath}/img_{model.algorithm_type}_{j}-{i}.png',
                          aHandles=[Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * (len(arguments) + 7),
                          aLabels=['', ' - Kwargs -', '\n'.join(fmt_kwargs), '', f'Dataset Size: {size}', f'Attacks: {n_attack}', f'Distribution: {data_dist}', f'Projection: {projection}'],
                          title=f'Gaussian Distribution ({model.algorithm_type}) :: #{i+1}')
        
        j += 1

if __name__ == "__main__":
    # gaussian_plots(runs=10, distribution=(0, 1), labeler=lambda X: [1 if x[0] < 0 else -1 for x in X],
    #          samples=40, types=['l1','l2','elastic'], num_attack=4, args=[{'epsilon':1e-03, 'sigma':1e-03}, {'epsilon':1e-05, 'sigma':1e-05}],
    #          projection=4, path='images/proj4')
    # gaussian_plots(runs=10, distribution=(0, 1), labeler=lambda X: [1 if x[0] < 0 else -1 for x in X],
    #          samples=40, types=['l1','l2','elastic'], num_attack=4, args=[{'epsilon':1e-03, 'sigma':1e-03}, {'epsilon':1e-05, 'sigma':1e-05}],
    #          projection=10, path='images/proj10')
    
    # write_comparative_rand('inp2.json', 200, 20, lambda X: [1 if x[0] < 10 else -1 for x in X], (10, 10), (-40, 60), distribution=True)
    # comparative_plots('inp1.json', center=(10, 10), args=[{}, {'sigma': 0.1}, {'epsilon': 0.1}, {'sigma': 1e-5}, {'epsilon': 1e-5}])
    # comparative_plots('inp2.json',
    #                   args=[{},
    #                         {'epsilon': 0.0125},
    #                         {'epsilon': 0.025},
    #                         {'epsilon': 0.05},
    #                         {'epsilon': 0.1},
    #                         {'epsilon': 0.2},
    #                         {'epsilon': 0.4},
    #                         {'epsilon': 0.8}])