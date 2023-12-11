#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:   cal corr among input files with col names by filenames    :
@Date     :2023/12/01 19:34:55
@Author      :Tingfeng Xu
@version      :1.0
'''
import argparse
from scipy.stats import pearsonr
import pandas as pd 
from pathlib import Path 
from multiprocessing import Pool
import math 
from functools import partial
import os 

import time 
import warnings
import textwrap
from tqdm import tqdm 

import statsmodels.api as sm


warnings.filterwarnings("ignore")
import time 
class Timing(object):
    """
    计时器
    """
    def __init__(self):
        self.start = time.time()

    def __call__(self):
        return time.time() - self.start 

    def __str__(self):
        return str(self.__call__())

    def __repr__(self):
        return str(self.__call__())

def cal_pearsonr(x, y):
    try:
        r, p = pearsonr(x, y)
    except:
        r, p = None, None 
    return {"r":r, "pvalue":p}
    

def logistic_regression(data, x, y, confounding=None):
    # Define the independent variables (X) and the dependent variable (y)
    if confounding:
        if not isinstance(confounding, list):
            confounding = [confounding]
    
    x_counfound = [x] + confounding if confounding else [x]

    # drop na 
    data = data.dropna(subset=x_counfound + [y], how="any").reset_index(drop=True)

    X = data[x_counfound]
    y = data[y]

    # Add a constant term to the independent variables
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.Logit(y, X)
    try:
        result = model.fit()

        # Get the beta coefficients
        beta = result.params

        # Get the p-values of the coefficients
        p_values = result.pvalues

        return {"beta": beta[x], "pvalue":p_values[x]}
    except:
        return {"beta": None, "pvalue":None}

def linear(data, x, y, confounding=None):
    # Define the independent variables (X) and the dependent variable (y)
    if confounding:
        if not isinstance(confounding, list):
            confounding = [confounding]
    
    x_counfound = [x] + confounding if confounding else [x]

    # drop na 
    data = data.dropna(subset=x_counfound + [y], how="any").reset_index(drop=True)

    X = data[x_counfound]
    y = data[y]

    # Add a constant term to the independent variables
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.OLS(y, X, )

    result = model.fit()

    # Get the beta coefficients
    beta = result.params

    # Get the p-values of the coefficients
    p_values = result.pvalues

    return {"beta": beta[x], "pvalue":p_values[x]}
    # except:
    #     return {"beta": None, "pvalue":None}



def cal_corrs(data, x, y, method, cond_cols=None):
    if method == "pearson":
        tmp_data = data[[x, y]].dropna()
        return cal_pearsonr(tmp_data[x], tmp_data[y])
    elif method == "logistic":
        return logistic_regression(data, x, y, cond_cols)
    elif method == "linear":
        return linear(data, x, y, cond_cols)
    else:
        return NotImplementedError(f"method {method} not supported yet")



def cross_corrs(main_df, key_cols, method="pearson", cond_cols=None):
    """
    main_df cols is equal: [query_cols, key_cols]
    so query_cols = main_df.columns - key_cols

    will combination between query_cols and key_cols 

    return: pd.DataFrame
    
    """
    res = []
    if isinstance(main_df, str): # read tmp_part_file_path
        main_df = read_data(main_df)

    query_cols = list(set(main_df.columns) - set(key_cols)) if cond_cols is None else list(set(main_df.columns) - set(key_cols) - set(cond_cols)) # main_df.columns - key_cols => query_cols 
    
    # total = len(query_cols) * len(key_cols)

    # pbar = tqdm(
    #         unit="it",
    #         total=total,
    #         desc=f"Cal corrs with {method} method",
    #     )

    for query_col in query_cols: # query 
        for key_col in key_cols: # key
            q_k_df = main_df[[query_col, key_col]].dropna()  # drop na

            # cal corrs 
            # corr_dict = cal_corrs(x = q_k_df[query_col], y= q_k_df[key_col], method=method)
            corr_dict = cal_corrs(data = q_k_df, x= query_col, y=key_col, method = method, cond_cols = cond_cols)
            res.append({**{"query":query_col, "key":key_col} , **corr_dict})
            # pbar.update(1)
    return pd.DataFrame(res) 

def read_data(path:str):
    """
    for pickle or csv files only 
    """
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    else:
        return pd.read_csv(path)

def get_name(x):
    return Path(x).name


def average_list(x, nums = 5):
    """
    a = range(10)
    split_list = average_list(a, 2)
    split_list => [[0, 1,2,3,4], [5,6,7,8,9]]

    """
    l = len(x)
    step = math.ceil(l/nums)
    return [x[i:i+step] for i in range(0, l ,step)]




def getParser():
    parser = argparse.ArgumentParser(
        prog = str(__file__), 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        %prog cross corrs 
        @Author: xutingfeng@big.ac.cn 

        Version: 1.0
        Exmaple:
            1. for linear --method linear ; q is x while k is y; will conbiantion x1 with y....x_n with y and only use --key_cols or all if no --key_cols
            cross_corr.py -q Olink_v2.pkl -k cad.pkl -o cad_olink/cad_olink_linear.csv --method linear -t 5 --key_cols ldl_a
            2. for logistic --method logistic; q is x while k is y; will conbiantion x1 with y....x_n with y and only use --key_cols or all if no --key_cols
            cross_corr.py -q Olink_v2.pkl -k cad.pkl -o cad_olink/cad_olink_logistic.csv --method logistic -t 5 --key_cols cad
            3. for only pearson corr --method pearson; q and k will cal pearson corr --key_cols will use selected cols or all if no --key_cols
        """
        ),
    )
    # main params
    parser.add_argument("-q", "--query", dest="query", help="query file path", required=True)
    parser.add_argument("-k", "--key", dest="key", help="key file path", required=True)
    parser.add_argument("-o", "--output", dest = "output", help = "outpu file name", required=True)
    parser.add_argument("-t", "--threads", dest="threads", help="processes of this ", default=5, type=int, required=False)

    # opt params
    parser.add_argument("--on", dest="index_on",
                         help="index cols to match query and key, defualt the common cols of query and key", 
                         required=False, default=[], nargs="+", type=int)
    parser.add_argument("-m", "--method", dest="method", default="pearson", required=False, choices=["pearson", "linear", "logistic"]) # may supported for multiple method
    parser.add_argument("--lowmem", action="store_true", dest="lowmem", help="low memory for cal")
    parser.add_argument("--cond", dest="cond_path", help="confounding file path, should be as same as q and k and used with --method linear or logistic", required=False)
    parser.add_argument("--cond_cols", dest= "cond_cols", help="confounding cols, should be in cond_path files", required=False, nargs="+", default=[])
    parser.add_argument("--key_cols", dest="key_cols", help="key cols to cal corrs, default all cols of key", required=False, nargs="+", default=[])
    return parser

if __name__ == "__main__":
    # input , currently only supported for two files 
    parser = getParser()
    args = parser.parse_args()

    query_path = args.query
    key_path = args.key
    output = args.output
    threads = args.threads

    index_cols = None if len(args.index_on) ==0 else args.index_on
    method = args.method
    lowmem = args.lowmem
    cond_path = args.cond_path
    cond_cols_used = args.cond_cols # [Age, Sex]....
    key_cols_used = args.key_cols
    timing = Timing()
    # confilict params check 
    if method not in ["logistic", "linear"]:
        if cond_path or len(cond_cols_used) > 0:
            raise ValueError("confounding only supported for logistic and linear method")

    # read data
    query_name, query_df = get_name(query_path), read_data(query_path)
    key_name ,key_df = get_name(key_path), read_data(key_path)

    query_cols, key_cols = query_df.columns, key_df.columns 
    
    # merge two df
    if index_cols:
        query_on = query_cols[index_cols]
        key_on = key_cols[index_cols]
        main_df = query_df.merge(key_df, left_on=query_on, right_on=key_on)

        on_cols = [query_on+key_on]

        query_cols = [col for col in query_cols if col not in query_on]
        key_cols = [col for col in key_cols if col not in key_on] if len(key_cols_used) == 0 else key_cols_used

    else:
        on_cols = list(set(query_cols).intersection(key_cols))
        main_df = query_df.merge(key_df)

        query_cols = [col for col in query_cols if col not in on_cols]
        key_cols = [col for col in key_cols if col not in on_cols] if len(key_cols_used) == 0 else key_cols_used

    # confounding with only logistic and linear
    if method in ["logistic", "linear"] and cond_path:
        cond_df = read_data(cond_path)
        cond_cols = cond_df.columns
        if index_cols:
            cond_on = cond_cols[index_cols]
        else:
            cond_on = list(set(main_df.columns).intersection(cond_cols))
        
        cond_cols = [col for col in cond_cols if col not in cond_on] if len(cond_cols_used) == 0 else cond_cols_used # 排除index col来得到最终使用的cond cols
        print(f"采用的矫正因素有：{','.join(cond_cols)}")
        main_df = main_df.merge(cond_df, left_on=on_cols, right_on=cond_on)
        del cond_df

        for rm_col in cond_on:  # drop confudounding index cols 
            if rm_col in main_df.columns:
                main_df.pop(rm_col)



    # del on_cols 
    for rm_col in on_cols: # TODO: on_cols 可能存在bug，需要测试
        if rm_col in main_df.columns:
            main_df.pop(rm_col)

    del key_df  # clear for memory

    # split by threads 
    if lowmem: # save to local tmp file and read while running
        parts_df = [] 
        tmp_dir = Path(output).parent
        print(f"--lowmem ， 采用低内存模式，将会保存中间文件到本地，可能会占用大量磁盘空间，保存在该路径下：{str(tmp_dir)}")

        for idx, part_cols in enumerate(average_list(query_cols, threads)):
            tmp_save_file = tmp_dir/f"tmp_part_{i}.pkl"
            if cond_path: # save with cond_cols
                main_df[part_cols + key_cols + cond_cols].to_pickle(str(tmp_save_file))
            else:
                main_df[part_cols + key_cols].to_pickle(str(tmp_save_file))
            parts_df.append(tmp_save_file)
    else:
        if cond_path:
            parts_df = [main_df[parts_col + key_cols + cond_cols].copy() for parts_col in average_list(query_cols, threads)]
        else:
            parts_df = [main_df[parts_col + key_cols].copy() for parts_col in average_list(query_cols, threads)]  # [part1_df, part2_df, ....]

    del query_df  # clear for memory 
    del main_df # clear for memory


    # TODO: support for regression method 
    if cond_path:
        cal_corrs_multiprocess = partial(cross_corrs, key_cols = key_cols, method=method, cond_cols=cond_cols)
    else:
        cal_corrs_multiprocess = partial(cross_corrs, key_cols = key_cols, method=method)

    with Pool(threads) as p: 
        # res = list(tqdm(p.imap(cal_corrs_multiprocess, parts_df), total=len(parts_df), desc="正在计算..."))
        p.map(cal_corrs_multiprocess, parts_df)
    corr_results_df = pd.concat(res).reset_index(drop=True)

    if lowmem: 
        for tmp_path in parts_df:
            os.remove(tmp_path)

    # save files 
    if output.endswith(".gz"):
        corr_results_df.to_csv(output, compression="gzip", index=False, na_rep="NA")
    else:
        corr_results_df.to_csv(output, index=False, na_rep="NA")

    print(f"总共消耗{timing():.2f}s")
