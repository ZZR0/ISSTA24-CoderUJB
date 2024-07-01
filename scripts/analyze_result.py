import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42	

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_task_result(models, task, analyze_keys=[], result_keys=[], output_folder="./", save_suffix="", comb_keys=[]):
    result_dict = {
        "model": []
    }
    
    if len(comb_keys) == 0:
        for analyze_key in analyze_keys:
            if "coverage" in analyze_key:
                comb_keys.append(analyze_key)
                result_dict[analyze_key] = []
            else:
                for result_key in result_keys:
                    comb_key = f"{analyze_key}|{result_key}"
                    comb_keys.append(comb_key)
                    result_dict[comb_key] = []
    else:
        for comb_key in comb_keys:
            result_dict[comb_key] = []
        
    for model in models:
        model_name, gen_mode = model
        result_file = os.path.join(FILE_DIR, "..", "log", model_name, task, f"evaluation-{save_suffix}-{gen_mode}.json")
        result = json.load(open(result_file))
        result_dict["model"].append(model_name)
        for cmb_key in comb_keys:
            print(model, task, cmb_key)
            if "|" in cmb_key:
                analyze_key, result_key = cmb_key.split("|")
                value = result[analyze_key][result_key]
                if "count" not in result_key and result_key != "error":
                    value = round(result[analyze_key][result_key]*100, 2)
            else:
                value = result[cmb_key]
                if "count" not in cmb_key and cmb_key != "error":
                    value = round(result[cmb_key]*100, 2)
            result_dict[cmb_key].append(value)
            
    os.makedirs(os.path.join(FILE_DIR, "..", "log", "_combine_result", output_folder), exist_ok=True)
    pd.DataFrame(result_dict).to_csv(
        os.path.join(FILE_DIR, "..", "log", "_combine_result", output_folder, f"{task}_result.csv"),
        index=False)


def draw_leida(models, tasks, print_key, output_folder="./", save_suffix="", save_file="leida.png"):
    # task_map = {
    #     "codeujbcomplete": "Functional Code Generation",
    #     "codeujbrepair": "Automated Program Repair",
    #     "codeujbtestgen": "Code-based Test Generation",
    #     "codeujbtestgenissue": "Issue-based Test Generation",
    #     "codeujbdefectdetection": "Defect Detection"
    # }
    task_map = {
        "codeujbcomplete": "FCG",
        "codeujbrepair": "APR",
        "codeujbtestgen": "CTG",
        "codeujbtestgenissue": "ITG",
        "codeujbdefectdetection": "DD"
    }
    
    model_map = {
        "codellama-13b": "CodeLlama-13B",
        "codellama-34b": "CodeLlama-34B",
        "wizardcoder-python-13b": "WizardCoder-Python-13B",
        "wizardcoder-python-34b": "WizardCoder-Python-34B",
        "claude-instant-1": "Claude-1",
        "gpt-3.5-turbo-0301": "GPT-3.5-Turbo",
        "gpt-4-0314": "GPT-4",
        
    }
    
    combine_result = {}
    for task in tasks:
        cmb_key = print_key[task]
        combine_result[task] = {}
        for model in models:
            model_name, gen_mode = model
            suffix = save_suffix[task] if isinstance(save_suffix, dict) else save_suffix
            result_file = os.path.join(FILE_DIR, "..", "log", model_name, task, f"evaluation-{suffix}-{gen_mode}.json")
            result = json.load(open(result_file))
            # print(result)
            if "|" in cmb_key:
                analyze_key, result_key = cmb_key.split("|")
                value = result[analyze_key][result_key]
                if "count" not in result_key and result_key != "error":
                    value = round(result[analyze_key][result_key]*100, 2)
            else:
                value = result[cmb_key]
                if "count" not in cmb_key and cmb_key != "error":
                    value = round(result[cmb_key]*100, 2)
            combine_result[task][model_name] = {print_key[task]: value}
    
    results = []
    legends = []
    for model in models:
        model_name, mode = model
        result = {
            task_map[task]: combine_result[task][model_name][print_key[task]]
            for task in tasks
        }
        results.append(result)
        legends.append(model_map[model_name])
    
    print(results)
    print(legends)
    labels = legends
    spoke_labels = list(results[0].keys())
    task_max_value = {key: max([items[key] for items in results]) for key in spoke_labels}
    case_data = [[0.8*items[key]/task_max_value[key] for key in spoke_labels] for items in results]
    
    # results = [{"大学英语": 87, "高等数学": 79, "体育": 95, "计算机基础": 92, "程序设计": 85},
    #         {"大学英语": 80, "高等数学": 90, "体育": 91, "计算机基础": 85, "程序设计": 88}]
    N = len(spoke_labels)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(4.5, 3), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    # colors = ['blue', 'orange', 'green', 'red', 'k', 'brown', ]
    colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    # colors = ['#1f78b4', '#a6cee3', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c']

    # Plot the four cases from the example data on separate axes
    # for ax, (title, case_data) in zip(axs.flat, data):
    # title = ""
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([])
    # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #                 horizontalalignment='center', verticalalignment='center')
    ax.plot(theta, [1, 0, 0, 0, 0], color=(176 / 255, 176 / 255, 176 / 255), 
            label='_nolegend_',
            linewidth=0.01, )
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(spoke_labels)
    
    # 每60度标注一个文本
    for idx, angle in enumerate(range(0, 360, 360//N)):
        ax.text(np.radians(angle), 0.85, str(round(list(task_max_value.values())[idx], 2)),
                horizontalalignment='center', verticalalignment='center', fontsize=10,
                rotation=angle)

    # add legend relative to top-left plot
    legend = ax.legend(labels, loc=(0.6, .85),
                              labelspacing=0.1, fontsize='small')

    # fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')
    # plt.legend(legends, bbox_to_anchor=(-0.1, 0.1), loc='lower left')

    plt.savefig(os.path.join(FILE_DIR, "../log/_combine_result", output_folder, save_file))


def analysis_table4_result():
    output_folder = "analysis_table4"
    models = [
        ("codellama-7b", "complete"),
        ("codellama-13b", "complete"),
        ("codellama-34b", "complete"),
        ("starcoderbase-15b", "complete"),
        
        ("codellama-python-7b", "complete"),
        ("codellama-python-13b", "complete"),
        ("codellama-python-34b", "complete"),
        ("starcoder-15b", "complete"),
        ("starcoder-java-15b", "complete"),
        
        ("codellama-instruct-7b", "chat"),
        ("codellama-instruct-13b", "chat"),
        ("codellama-instruct-34b", "chat"),
        
        ("wizardcoder-python-7b", "chat"),
        ("wizardcoder-python-13b", "chat"),
        ("wizardcoder-python-34b", "chat"),
        
        ("wizardcoder-15b", "chat"),
        
        ("claude-instant-1", "chat"),
        ("gpt-3.5-turbo-0301", "chat"),
        ("gpt-4-0314", "chat"),
    ]
    
    comb_keys = [
        "pass_syntax|pass@k-1", "pass_compile|pass@k-1", 
        "pass_all|pass@k-1", "pass_all|pass@k-10", 
        "pass_all|count"
    ]
    get_task_result(models, "codeujbcomplete", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    get_task_result(models, "codeujbrepair", comb_keys=comb_keys, 
                        output_folder=output_folder, save_suffix="default|2048")
    
    comb_keys = [
        "pass_syntax|pass@k-1", "pass_compile|pass@k-1", 
        "pass_trigger|pass@k-1", "pass_trigger|pass@k-10", 
        "pass_trigger|count", "diff_coverage"
    ]
    get_task_result(models, "codeujbtestgen", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    get_task_result(models, "codeujbtestgenissue", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    comb_keys = [
        "results_one|error", 
        "results_one|acc", "results_one|acc_w_error", 
        "results_one|precision", "results_one|recall", "results_one|f1", 
    ]
    get_task_result(models, "codeujbdefectdetection", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    # tasks = [
    #     "codeujbcomplete", "codeujbrepair", "codeujbtestgen", "codeujbtestgenissue",
    # ]
    # print_key = {
    #     "codeujbcomplete":"pass_all|pass@k-1",
    #     "codeujbrepair":"pass_all|pass@k-1",
    #     "codeujbtestgen":"pass_trigger|pass@k-1",
    #     "codeujbtestgenissue":"pass_trigger|pass@k-1",
    # }
    # draw_leida(models, tasks, print_key, output_folder="analysis_rq1")


def analysis_table5_result():
    output_folder = "analysis_table5"
    models = [
        ("codellama-7b", "complete"),
        ("codellama-13b", "complete"),
        ("codellama-34b", "complete"),
        
        ("codellama-instruct-7b", "chat"),
        ("codellama-instruct-13b", "chat"),
        ("codellama-instruct-34b", "chat"),
    ]
    
    comb_keys = [
        "pass_syntax|pass@k-1", "pass_compile|pass@k-1", 
    ]
    get_task_result(models, "codeujbcomplete", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    comb_keys = [
        "pass_syntax|pass@k-1", "pass_compile|pass@k-1", 
    ]
    get_task_result(models, "codeujbtestgen", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    get_task_result(models, "codeujbtestgenissue", comb_keys=comb_keys,
                        output_folder=output_folder, save_suffix="default|2048")
    
    

def get_model_result(model, tasks, keys_map, suffix_list, output_folder="./"):
    model_name, gen_mode = model
    
    result_dict = {
        "task": [],
        "metric": [],
    }
    for task in tasks:
        for analyze_key in keys_map[task]:
            result_dict["task"].append(task)
            result_dict["metric"].append(analyze_key)
    
    for suffix in suffix_list:
        result_dict[suffix] = []
        
        for task in tasks:
            result_file = os.path.join(FILE_DIR, "..", "log", model_name, task, f"evaluation-{suffix}-{gen_mode}.json")
            result = json.load(open(result_file))
            
            for cmb_key in keys_map[task]:
                print(model, task, cmb_key)
                if "|" in cmb_key:
                    analyze_key, result_key = cmb_key.split("|")
                    value = result[analyze_key][result_key]
                    if "count" not in result_key and result_key != "error":
                        value = round(result[analyze_key][result_key]*100, 2)
                else:
                    value = result[cmb_key]
                    if "count" not in cmb_key and cmb_key != "error":
                        value = round(result[cmb_key]*100, 2)
                result_dict[suffix].append(value)
            
    os.makedirs(os.path.join(FILE_DIR, "..", "log", "_combine_result", output_folder), exist_ok=True)
    pd.DataFrame(result_dict).to_csv(
        os.path.join(FILE_DIR, "..", "log", "_combine_result", output_folder, f"{model_name}_result.csv"),
        index=False)


def analysis_table3_result():
    output_folder = "analysis_table3"
    models = [
        ("codellama-13b", "complete"),
        ("starcoderbase-15b", "complete"),
    ]
    
    tasks = [
        "codeujbcomplete", "codeujbtestgen", "codeujbrepair", "codeujbdefectdetection"
    ]
    
    comb_keys_map = {
        "codeujbcomplete":["pass_all|pass@k-1", "pass_all|pass@k-10", "pass_all|count"],
        "codeujbtestgen":["pass_trigger|pass@k-1", "pass_trigger|pass@k-10", "pass_trigger|count", "diff_coverage"],
        "codeujbrepair":["pass_all|pass@k-1", "pass_all|pass@k-10", "pass_all|count"],
        "codeujbdefectdetection":["results_one|acc_w_error", "results_one|error"],
    }
    
    suffix_list = ["default|2048", "fs1|2048", "fs4|2048",]
    
    for model in models:
        get_model_result(model, tasks, comb_keys_map, suffix_list,
                        output_folder=output_folder)


def analysis_figure4_result():
    output_folder = "analysis_figure4"
    models = [
        ("gpt-4-0314", "chat"),
        ("gpt-3.5-turbo-0301", "chat"),
        ("claude-instant-1", "chat"),
        ("wizardcoder-python-34b", "chat"),
        ("codellama-34b", "complete"),
        # ("codellama-13b", "complete"),
    ]
    
    tasks = [
        "codeujbcomplete", "codeujbtestgen", "codeujbtestgenissue", "codeujbrepair", "codeujbdefectdetection"
    ]
    print_key = {
        "codeujbcomplete":"pass_all|pass@k-1",
        "codeujbrepair":"pass_all|pass@k-1",
        "codeujbtestgen":"pass_trigger|pass@k-1",
        "codeujbtestgenissue":"pass_trigger|pass@k-1",
        "codeujbdefectdetection":"results_one|acc_w_error",
    }
    save_suffix_map = {
        "codeujbcomplete":"default|2048",
        "codeujbrepair":"default|2048",
        "codeujbtestgen":"default|2048",
        "codeujbtestgenissue":"default|2048",
        "codeujbdefectdetection":"default|2048",
    }
    draw_leida(models, tasks, print_key, output_folder=output_folder, 
               save_suffix=save_suffix_map, save_file="leida_pass1_fixed.pdf")
    
    print_key = {
        "codeujbcomplete":"pass_all|pass@k-10",
        "codeujbrepair":"pass_all|pass@k-10",
        "codeujbtestgen":"pass_trigger|pass@k-10",
        "codeujbtestgenissue":"pass_trigger|pass@k-10",
        "codeujbdefectdetection":"results_one|acc_w_error",
    }
    draw_leida(models, tasks, print_key, output_folder=output_folder, 
               save_suffix="default|2048", save_file="leida_pass10_fixed.pdf")


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == "__main__":
    analysis_table3_result()
    analysis_figure4_result()
    analysis_table4_result()
    analysis_table5_result()

