import warnings

warnings.filterwarnings("ignore")

from pBench import pBench_fast
import poisoning
import json
import argparse
import time
import os
import csv
import pandas as pd
from datetime import datetime

# gets the command line arguments for the benchmarks
def get_arguments():
    parser = argparse.ArgumentParser(description="Run benchmarks for poisoning.")
    parser.add_argument(
        "argfile", help="Provides information about running benchmarks."
    )
    parser.add_argument("-o", "--out", default=None, help="Output filename.")
    parser.add_argument(
        "-p",
        "--prefix",
        action="store_true",
        help="Prefix output filename with date and time.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print extra output."
    )
    parser.add_argument(
        "-b", "--blind", action="store_true", help="Don't show the progress bar."
    )
    parser.add_argument(
        "-a", "--append", action="store_true", help="Append to file if possible."
    )

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.splitext(os.path.basename(args.argfile))[0]

    return args


# reads json file contianing the information about how to run the benchmark
def read_argfiles(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return [read_argfile(d) for d in data]


def read_argfile(data):
    types = {
        "xiao": [poisoning.xiao2018],
        "fred": [poisoning.frederickson2018],
        "both": [poisoning.xiao2018, poisoning.frederickson2018],
    }

    data["model"] = types[data["model"]]

    return data


def get_data(dataset, attacks):
    extension = os.path.splitext(dataset)[1]
    if extension == ".csv":
        dataset = pd.read_csv(dataset, sep=",", header=None)
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values
    elif extension == ".json":
        with open(dataset, "r") as f:
            X, Y = json.load(f)
    else:
        raise ValueError(f"Cannot accept {extension} files.")

    if attacks.get("range", False):
        tp = "range"
        Attacks = X[attacks["range"][0] : attacks["range"][1]]
        Labels = Y[attacks["range"][0] : attacks["range"][1]]
    elif attacks.get("percent", False):
        tp = "percent"
        Attacks = int(len(X) * (attacks["percent"] / 100))
        Labels = None
    elif attacks.get("auto", False):
        tp = "auto"
        Attacks = attacks["auto"]
        Labels = None
    else:
        raise TypeError("Invalide attack type.")

    return X, Y, Attacks, Labels, (len(X), len(X[0])), tp


def class_to_string(class_type):
    return "xiao2018" if class_type == poisoning.xiao2018 else "frederickson2018"


# run the benchmark by creating the necessary class and running the correct type of function, timing it and returning time
# convert information into data that can be written into csv
def run_benchmark(class_type, dataset, attacks, projection, arguments):

    X, Y, Attacks, Labels, dataset_size, tp = get_data(dataset, attacks)

    if tp == "auto":
        attack_size = Attacks
        with pBench_fast() as bench:
            model = class_type(**arguments)
            _, Attacks = model.autorun(X, Y, Attacks, projection, rInitial=True)
    elif tp == "percent":
        attack_size = Attacks
        with pBench_fast() as bench:
            model = class_type(**arguments)
            _, Attacks = model.autorun(X, Y, Attacks, projection, rInitial=True)
    elif tp == "range":
        with pBench_fast() as bench:
            model = class_type(**arguments)
            model.run(X, Y, Attacks, Labels, projection)
        attack_size = len(Attacks)

    return (
        os.path.splitext(os.path.basename(dataset))[0],
        class_to_string(class_type),
        model.algorithm_type,
        dataset_size[0],
        dataset_size[1],
        len(Attacks),
        projection,
        bench.get_seconds(),
        arguments,
        attacks,
        bench.get_time(),
    )


class bench_results:
    def __init__(self, filename, prefix=False, append=False):
        time_prefix = datetime.now().strftime("[%Y-%m-%d][%H.%M.%S]") + "_"
        self.filename = (time_prefix if prefix else "") + filename

        if append:
            self.flag = "a"
        else:
            self.flag = "w"

        self.file = None
        self.writer = None
        self.columns = [
            "dataset",
            "implementation",
            "algorithm_type",
            "n_samples",
            "n_features",
            "n_attacks",
            "projection",
            "seconds",
            "argument_info",
            "attack_info",
            "time_formatted",
        ]

    def __enter__(self):
        self.file = open(self.filename + ".csv", self.flag, newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.columns)
        self.file.flush()
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def add(self, data):
        self.writer.writerow(data)
        self.file.flush()

    def add_separator(self):
        self.writer.writerow(["" for item in self.columns])
        self.file.flush()


class no_bar:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


def main():

    args = get_arguments()
    data_array = read_argfiles(args.argfile)
    f_path = os.path.dirname(os.path.realpath(args.argfile))

    with bench_results(args.out, args.prefix, args.append) as results:
        for data in data_array:

            bar_iterations = (
                len(data["datasets"])
                * len(data["model"])
                * len(data["attacks"])
                * len(data["model_args"])
                * len(data["projections"])
                * data["iter"]
            )

            if not args.blind:
                import alive_progress

                progress_bar = alive_progress.alive_bar
            else:
                progress_bar = no_bar

            with progress_bar(
                bar_iterations, enrich_print=False, bar="classic2", spinner="classic"
            ) as bar:
                for i in range(data["iter"]):
                    for type_args in data["model_args"]:
                        for dataset in data["datasets"]:
                            for type in data["model"]:
                                for projection in data["projections"]:
                                    for attack in data["attacks"]:
                                        if args.verbose:
                                            impl = class_to_string(type)
                                            data_name = os.path.splitext(
                                                os.path.basename(dataset)
                                            )[0]
                                            print(
                                                f"Running: {{Implementation: {impl} | Dataset: {data_name} | Projection: {projection} | Attack: {attack}}}"
                                            )
                                        result = run_benchmark(
                                            type,
                                            os.path.join(f_path, dataset),
                                            attack,
                                            projection,
                                            type_args,
                                        )
                                        results.add(result)
                                        if args.verbose:
                                            print(f"Done in {result[7]} seconds.")
                                        bar()

            results.add_separator()


if __name__ == "__main__":
    main()
