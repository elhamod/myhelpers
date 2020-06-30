import pandas as pd
from IPython.display import display, HTML
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import json
import hashlib

import statistics
# import qgrid
import matplotlib.pyplot as plt
from pivottablejs import pivot_ui

from .confusion_matrix_plotter import plot_confusion_matrix, generate_classification_report

aggregateStatFileName = "agg_experiments.csv"
rawStatFileName = "raw_experiments.csv"
confusionMatrixParamsFileName="params.json"
confusionMatrixFileName = "confusion_matrix.pdf"


# Given a confusion matrix, gets metris of fine with respect to coarse
class fine_Coarse_Statistics:
    def __init__(self, cm, dataset):
        self.dataset = dataset
        self.cm = cm
    
    def get_statistics(self, fine_index):
        true_positives = self.cm[fine_index, fine_index]
        fine_name = self.dataset.getfineOfIndex(fine_index)
        fine_names = self.dataset.getfineWithinCoarse(self.dataset.getCoarseFromfine(fine_name))
        fine_indexes = list(map(lambda x: self.dataset.getfineList().index(x), fine_names))

        within_coarse_FP = np.sum(self.cm[fine_indexes, fine_index]) - true_positives
        out_of_coarse_FP = np.sum(self.cm[:, fine_index]) - true_positives - within_coarse_FP
        within_coarse_FN = np.sum(self.cm[fine_index, fine_indexes]) - true_positives
        out_of_coarse_FN = np.sum(self.cm[fine_index, :]) - true_positives - within_coarse_FN
        return {
            "TP": true_positives,
            "FP_within_coarse": within_coarse_FP,
            "FP_out_of_coarse": out_of_coarse_FP,
            "FN_within_coarse": within_coarse_FN,
            "FN_out_of_coarse": out_of_coarse_FN,
        }

    def get_precision_recall(self, fine_index):
        fine_statistics = self.get_statistics(fine_index)
        
        TP = fine_statistics["TP"]
        FP_within_coarse = fine_statistics["FP_within_coarse"]
        FP_out_of_coarse = fine_statistics["FP_out_of_coarse"]
        FN_within_coarse = fine_statistics["FN_within_coarse"]
        FN_out_of_coarse = fine_statistics["FN_out_of_coarse"]
        
        Precision_within_coarse = TP/(TP + FP_within_coarse) if (TP + FP_within_coarse) != 0 else 0
        Precision_out_of_coarse = TP/(TP + FP_out_of_coarse) if (TP + FP_out_of_coarse) != 0 else 0
        Recall_within_coarse = TP/(TP + FN_within_coarse) if (TP + FN_within_coarse) != 0 else 0
        Recall_out_of_coarse = TP/(TP + FN_out_of_coarse) if (TP + FN_out_of_coarse) != 0 else 0
        Precision = TP/(TP + FP_within_coarse + FP_out_of_coarse) if (TP + FP_within_coarse + FP_out_of_coarse) != 0 else 0
        Recall = TP/(TP + FN_within_coarse + FN_out_of_coarse) if (TP + FN_within_coarse + FN_out_of_coarse) != 0 else 0
        
        return {
            "Precision_within_coarse": Precision_within_coarse,
            "Precision_out_of_coarse": Precision_out_of_coarse,
            "Recall_within_coarse": Recall_within_coarse,
            "Recall_out_of_coarse": Recall_out_of_coarse,
            "Precision": Precision,
            "Recall": Recall,
        }
    
    def get_F1Scores(self, fine_index):
        precision_recall_stats = self.get_precision_recall(fine_index)
        
        within_coarse_stats = [precision_recall_stats["Precision_within_coarse"], precision_recall_stats["Recall_within_coarse"]]
        f1_macro_within_coarse = statistics.harmonic_mean(within_coarse_stats)
        
        out_of_coarse_stats = [precision_recall_stats["Precision_out_of_coarse"], precision_recall_stats["Recall_out_of_coarse"]]
        f1_macro_out_of_coarse = statistics.harmonic_mean(out_of_coarse_stats)
        
        overall_stats = [precision_recall_stats["Precision"], precision_recall_stats["Recall"]]
        f1_macro = statistics.harmonic_mean(overall_stats)
        
        return {
            "f1_macro_within_coarse": f1_macro_within_coarse,
            "f1_macro_out_of_coarse": f1_macro_out_of_coarse,
            "f1_macro": f1_macro,
        }

# Given a confusion matrix, get statistics of coarse labels
class Coarse_Statistics:
    def __init__(self, cm, dataset):
        self.dataset = dataset
        self.cm = cm
    
    def get_statistics(self, coarse_index):
        true_positives = self.cm[coarse_index, coarse_index]

        FP = np.sum(self.cm[:, coarse_index]) - true_positives
        FN = np.sum(self.cm[coarse_index, :]) - true_positives
        return {
            "TP": true_positives,
            "FP": FP,
            "FN": FN,
        }

    def get_precision_recall(self, coarse_index):
        fine_statistics = self.get_statistics(coarse_index)
        
        TP = fine_statistics["TP"]
        FP = fine_statistics["FP"]
        FN = fine_statistics["FN"]
        
        Precision = TP/(TP + FP) if (TP + FP) != 0 else 0
        Recall = TP/(TP + FN) if (TP + FN) != 0 else 0
        
        return {
            "Precision": Precision,
            "Recall": Recall,
        }
    
    def get_F1Scores(self, coarse_index):
        precision_recall_stats = self.get_precision_recall(coarse_index)
        
        overall_stats = [precision_recall_stats["Precision"], precision_recall_stats["Recall"]]
        f1_macro = statistics.harmonic_mean(overall_stats)
        
        return {
            "f1_macro": f1_macro,
        }


# Allows for addin trial results and then aggregating them
class TrialStatistics:
    def __init__(self, experiment_name, prefix=None):
        self.df = pd.DataFrame()
        self.agg_df = pd.DataFrame()
        self.experiment_name = experiment_name
        
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)
            
        self.prefix = prefix
        self.trial_params_keys = []
        self.trial_results_keys = []
        
        self.confusionMatrices = {}
        self.agg_confusionMatrices = {}

    # Adds the parameters and results of a trial
    def addTrial(self, trial_params, trial_results, trial=None):
        # Reset aggregate information
        self.agg_df = pd.DataFrame()
        
        # preprocess
        trial_params_copy = self.preProcessParameters(trial_params)
        row_hash = {'experimentHash': getTrialName(trial_params_copy)}
        trial_params_with_hash = {**trial_params_copy, **row_hash}
        row_information = {**trial_params_copy, **trial_results}
        row_information = {**row_information, **row_hash}
        
        # Augment row information
        if trial is not None:
            row_information["trial"] = trial
            row_information["trialHash"] = getTrialName(trial_params_copy, trial)

        # Add row
        self.df = self.df.append(pd.DataFrame(row_information, index=[0]), ignore_index = True)

        # populate param and result lists keys
        for key in trial_params_with_hash:
            if key not in self.trial_params_keys:
                self.trial_params_keys.append(key)
        
        for key in trial_results:
            if key not in self.trial_results_keys:
                self.trial_results_keys.append(key)
                
    # Adds the predictions for confusion matrix calculations
    def addTrialPredictions(self, trial_params, predlist, lbllist, numberOffine):
        self.agg_confusionMatrices = {}
        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy(), labels = range(numberOffine))
        trial_params_copy = self.preProcessParameters(trial_params)
        trial_hash = getTrialName(trial_params_copy)
        if trial_hash not in self.confusionMatrices:
            self.confusionMatrices[trial_hash] = []
        self.confusionMatrices[trial_hash].append(conf_mat)
   
    def aggregateTrials(self):        
        # group by trial params
        groupedBy_df = self.df.groupby(self.trial_params_keys)
        
        # For each result key, calculate mean and std
        for key in self.trial_results_keys:
            groupedBy_df_summaried = groupedBy_df.agg({key:['mean','std']})
            self.agg_df = pd.concat([self.agg_df,groupedBy_df_summaried], axis=1, ignore_index=False)
            
        self.agg_df = self.agg_df.round(3).reset_index()
        
    def saveStatistics(self, aggregated=True):
        if aggregated:
            file_name = aggregateStatFileName
        else:
            file_name = rawStatFileName
            
        if self.prefix is not None:
            file_name = self.prefix + "_" + file_name
            
        if aggregated:
            if self.agg_df.empty:
                self.aggregateTrials()
            self.agg_df.to_csv(os.path.join(self.experiment_name, file_name))
        else:
            self.df.to_csv(os.path.join(self.experiment_name, file_name))  
        
    def showStatistics(self, aggregated=True, saveHTML=False):
        df = self.df.copy()
        if aggregated:
            if self.agg_df.empty:
                self.aggregateTrials()
            df = self.agg_df.copy()
            
        name = "aggregated statistics" if aggregated else "raw statistics"
        name_html = name+'.html'
#         df.columns = [' '.join(col).strip() for col in df.columns.values] # work around:https://github.com/quantopian/qgrid/issues/18#issuecomment-149321165
#         return qgrid.show_grid(df, show_toolbar=True)
        display(HTML(df.to_html()))
        if saveHTML:
            pivot_ui(df,outfile_path=os.path.join(self.experiment_name, name_html))
#         display(HTML(self.experiment_name+"/"+name_html))
            
    def getStatistic(self, trial_params, metric, statistic):
        if self.agg_df.empty:
            self.aggregateTrials()
        trial_params_copy = self.preProcessParameters(trial_params)
        row = self.agg_df.loc[self.agg_df['experimentHash'] == getTrialName(trial_params_copy)]
        return row[self.trial_results_keys][(metric, statistic)].item()
    
    
    
    

        
    # prints aggregate confusion matrix for trials    
    def printTrialConfusionMatrix(self, trial_params, lst, printOutput=False):
        aggregatePath = os.path.join(self.experiment_name, getTrialName(trial_params))
        if not os.path.exists(aggregatePath):
            os.makedirs(aggregatePath)

        file_name = confusionMatrixFileName
        if self.prefix is not None:
            file_name = self.prefix + "_" + file_name

        j = json.dumps(trial_params)
        f = open(os.path.join(aggregatePath, confusionMatrixParamsFileName),"w")        
        f.write(j)
        f.close() 
            
        return plot_confusion_matrix(self.getTrialConfusionMatrix(trial_params),
                                  lst,
                                  aggregatePath,
                                  file_name,
                                  printOutput)
    
    def aggregateTrialConfusionMatrices(self):
        for hash_key in self.confusionMatrices:
            confusionMatricesForHash = self.confusionMatrices[hash_key]
            self.agg_confusionMatrices[hash_key] = np.mean(confusionMatricesForHash, axis=0) 
        
    def prepareConfusionMatrix(self, trial_params):
        if not self.agg_confusionMatrices:
            self.aggregateTrialConfusionMatrices()
    
    def getTrialConfusionMatrix(self, trial_params):
        self.prepareConfusionMatrix(trial_params)
        
        return self.agg_confusionMatrices[getTrialName(trial_params)]
    

    
    
    def printF1table(self, trial_params, dataset):
        cm = self.getTrialConfusionMatrix(trial_params)

        if self.prefix == "coarse":
            columns = ['coarse', 'F1']
        else:
            columns = ['fine', 'coarse', 'F1']
#             if trial_params['useHeirarchy']:
            columns = columns + ['F1_within_coarse', 'F1_out_of_coarse']
                
        df = pd.DataFrame(columns=columns)

        if self.prefix == "coarse":
            stats = Coarse_Statistics(cm, dataset)
            for coarse_name in dataset.getCoarseList():
                coarse_index = dataset.getCoarseList().index(coarse_name)
                coarse_stats = stats.get_F1Scores(coarse_index)
                df.loc[coarse_index] = [" ".join([str(coarse_index), coarse_name]),
                                   coarse_stats["f1_macro"]]
        else:
            stats = fine_Coarse_Statistics(cm, dataset)
            for fine in range(len(dataset.getfineList())):
                fine_stats = stats.get_F1Scores(fine)
                fine_name = dataset.getfineOfIndex(fine)
                coarse_name = dataset.getCoarseFromfine(fine_name)
                coarse_index = dataset.getCoarseList().index(coarse_name)
                vals = [" ".join([str(fine), str(fine_name)]),
                                   " ".join([str(coarse_index), str(coarse_name)]),
                                   fine_stats["f1_macro"],]
#                 if trial_params['useHeirarchy']:
                vals = vals + [ fine_stats["f1_macro_within_coarse"],
                               fine_stats["f1_macro_out_of_coarse"]]
                df.loc[fine] = vals
            
        display(HTML(df.to_html()))
        file_name = "F1_Scores"
        if self.prefix == "coarse":
            file_name = file_name + "_coarse"
        pivot_ui(df,outfile_path=os.path.join(self.experiment_name, file_name+".html"))
    
    
    
    def trialScatter(self, x, y, aggregatedBy=None, save_plot=False):
        df = self.df
            
        file_name = 'plot ' + y + " to " +  x + ((' by ' + aggregatedBy) if aggregatedBy is not None else '')
                 
        
        # get unique values for aggregate by
        uniqueValues=['all']
        if aggregatedBy is not None:
            uniqueValues=df[aggregatedBy].unique()       

        # prepare axis
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        
                     
        for val in uniqueValues:
            if aggregatedBy:
                x_values = df.loc[df[aggregatedBy] == val][x].values
                y_values = df.loc[df[aggregatedBy] == val][y].values    
            else:
                x_values = df[x].values
                y_values = df[y].values

            im = ax.scatter(x=x_values,
                          y=y_values,
                          label=val)
            ax.legend()
            
        if save_plot:
            fig.savefig(os.path.join(self.experiment_name, file_name+".pdf"))

    
    def preProcessParameters(self, trial_params):
        trial_params_copy = {**trial_params, **{}}
        return trial_params_copy

def getTrialName(trial_params, trial_number=None):
    trialName = str(trial_params)
    if trial_number is not None:
        trialName = trialName + str(trial_number)
    return hashlib.sha224(trialName.encode('utf-8')).hexdigest()