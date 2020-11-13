import os, csv, shutil, tempfile, json
import random

def change_last_column(path, **kwargs):
    
    verbose = kwargs.pop('verbose', False)
    change = kwargs.pop('change', None)
    
    if kwargs:
        raise TypeError('Unknown inputs: ' + ', '.join(kwargs.keys()))
    
    if verbose:
        print(f'Looking for files in {path}.')
    
    for filename in os.listdir(path):
        if verbose:
            print(f'Opening file {filename} for reading.')
        with open(os.path.join(path, filename), newline='') as file:
            reader = csv.reader(file, delimiter=',')
            lines = list(reader)
        if verbose:
            print(f'Opening file {filename} for writing.')
        with open(os.path.join(path, filename), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            if verbose:
                    print(f'Changing file {filename}.')
            if change is None:
                lines = [line if int(line[-1]) != 0 else line[:-1] + ['-1'] for line in lines]
            else:
                if verbose:
                    print(f'Changing file {filename} with custom change.')
                lines = change(lines)
            if verbose:
                    print(f'Writing new file {filename}.')
            writer.writerows(lines)
            
    if verbose:
        print(f'Done changing files in {path}.')

def find_files(path, **kwargs):
    
    min_features = kwargs.pop('min_features', 20)
    min_samples = kwargs.pop('min_samples', None)
    label_amount = kwargs.pop('label_amount', 2)
    copy_path = kwargs.pop('copy_path', None)
    verbose = kwargs.pop('verbose', False)
    
    if kwargs:
        raise TypeError('Unknown inputs: ' + ', '.join(kwargs.keys()))
    
    files = []
    if verbose:
        print(f'Looking for files in {path}.')
    for filename in os.listdir(path):
        if not filename.endswith('.csv'):
            continue
        if verbose:
            print(f'Opening file {filename}.')
        with open(os.path.join(path, filename), newline='') as file:
            reader = csv.reader(file, delimiter=',')
            last = []
            num_samples = 0
            for row in reader:
                last.append(row[-1])
                num_samples += 1
            feature_size = len(row) - 1
            num_labels = len(set(last))
            print(set(last))

            if verbose:
                print(f'Finished reading {filename} :: Found {feature_size} features, {num_samples} samples, and {num_labels} labels.')

        if num_labels == label_amount and feature_size >= min_features and (num_samples >= min_samples if min_samples is not None else True):
            files.append(filename)
            if verbose:
                print(f'{filename} passed requirements.')
        else:
            if verbose:
                print(f'{filename} did not pass requirements.')
        if verbose:
                print(f'')

    if copy_path is not None:
        if verbose:
                print(f'{copy_path} was provided for copying to.')
        if (os.path.exists(copy_path)):
            if verbose:
                print(f'Removing existing {copy_path}.')
            tmp = tempfile.mktemp(dir=os.path.dirname(copy_path))
            shutil.move(copy_path, tmp)
            shutil.rmtree(tmp)
            
        if verbose:
                print(f'Making folder {copy_path}.')
        os.makedirs(copy_path)
        for file in files:
            if verbose:
                print(f'Copying {file}.')
            shutil.copy(os.path.join(path, file), copy_path)
            
    if verbose:
        print(f'\nFinished finding files.')
    
    return files

def make_test_inputs(path, **kwargs):
    
    destination = kwargs.pop('dest', os.getcwd())
    verbose = kwargs.pop('verbose', False)

    if verbose:
        print(f'Starting making tests at {path}.')

    def make_attacks(Matrix):
        Attacks = random.sample(Matrix, 5)
        Labels = [row[-1] for row in Attacks]
        Attacks = [row[:-1] for row in Attacks]

        return Attacks, Labels

    if kwargs:
        raise TypeError('Unknown inputs: ' + ', '.join(kwargs.keys()))

    for filename in os.listdir(path):
        if verbose:
            print(f'Making test for {filename}.')
        with open(os.path.join(path, filename), newline='') as file:
            reader = csv.reader(file, delimiter=',')
            lines = list(reader)
        
        lines = [[float(num) for num in line] for line in lines]
        
        X = [[num for num in line[:-1]] for line in lines]
        Y = [line[-1] for line in lines]
        
        Attacks, Labels = make_attacks(lines)
        
        with open(os.path.join(destination, f'Input_dataset_{filename.rstrip(".csv")}.json'), 'w') as outfile:
            if verbose:
                print(f'Making Input_dataset_{filename.rstrip(".csv")}.json.')
            json.dump([X, Y, Attacks, Labels], outfile)

    if verbose:
            print(f'Done writing tests.')

def run(from_path, to_path, **kwargs):
    
    download = kwargs.pop('download', None)
    check_raw = kwargs.pop('check_raw', False)
    write_links = kwargs.pop('write_link', False)
    change = kwargs.pop('change', check_raw)
    write_tests = kwargs.pop('write_tests', change)
    verbose = kwargs.pop('verbose', False)
    
    if download is not None:
        os.system(f'cd {from_path}')
        os.system(f'./download.sh {download}')
        os.system('cd ..')
    if check_raw:
        files = find_files(os.path.join(os.getcwd(), from_path), copy_path=to_path, verbose=verbose)
    if write_links:
        if verbose:
            print(f'Writing links file {from_path}/good_links.txt.')
        with open(os.path.join(os.getcwd(), f'{from_path}/good_links.txt'), 'w') as f:
            f.writelines(['https://github.com/gditzler/UA-ECE523-EngrAppMLData/blob/master/data/' + file + '\n' for file in files])
    if change:
        change_last_column(os.path.join(os.getcwd(), to_path), change=lambda lines: [line if int(line[-1]) != 0 else line[:-1] + ['-1'] for line in lines], verbose=verbose)
    if write_tests:
        make_test_inputs(os.path.join(os.getcwd(), to_path), dest=os.path.join(os.getcwd(), '../tests/inputs'), verbose=verbose)

if __name__ == '__main__':
    run('raw_data', 'good_data', verbose=True, check_raw=True)