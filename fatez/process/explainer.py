#!/usr/bin/env python3
"""
Interpret machine learning models

author: jy
"""



import re
import shap
import numpy as np
import pandas as pd
from warnings import warn
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance



class Explain:
    """
    Explain how machine learning models make decisions respectively.
    """
    def __init__(self, model, background_data = None):
        super(Explain, self).__init__()
        # make background example based on mean of every sample
        # ToDo:
        # Background generation may need to be revised
        # We may just use grn generated based on universal exp matrix
        self.model = model

        if background_mode == 'all':
            bases = trainer_data.all_data
        elif background_mode == 'mean':
            bases = pd.DataFrame().append(
                trainer_data.all_data.mean(axis = 0),
                ignore_index = True
            )
        else:
            raise lib.Error('Unknown Background Data Mode')
        self.result = self.__interpret_process(trainer_data, bases)
        self.result = self.__subtract_feature_score(self.result)

    # Calculate importances of each feature
    def __interpret_process(self, trainer_data, bases):
        shap_explainer = SHAP_Explainer(bases)
        feature_score_sum = None
        # sumFIs = None
        for record in trainer_data.models:
            # get model and data to explain
            print('     Interpreting:',record.model.id)
            print('         Accuracy:',record.accuracy)
            print('         AUROC:',record.auroc)
            print('         Loss:',float(record.loss))
            # Handling RFC cases
            if record.model.model_type == 'RFC':
                feature_score = shap_explainer.get_tree_explain(
                    record.model.clf,
                    trainer_data.all_data.iloc[record.correct_predictions,:]
                )
            # Handling GNB cases
            elif record.model.model_type == 'GNB':
                feature_score = shap_explainer.get_kernel_explain(
                    record.model.clf.predict_proba,
                    trainer_data.all_data.iloc[record.correct_predictions,:]
                )
            # Handling Logistic Regression cases
            elif record.model.model_type == 'Logit':
                feature_score = softmax(abs(record.model.clf.coef_[0]))
            # Handling SVM cases
            elif record.model.model_type == 'SVM':
                # Handle linear kernel SVC here
                if record.model.param['kernel'] == 'linear':
                    feature_score = softmax(abs(record.model.clf.coef_[0]))
                # Handle other cases here
                else:
                    feature_score = shap_explainer.get_kernel_explain(
                        record.model.clf.predict_proba,
                        trainer_data.all_data.iloc[record.correct_predictions,:]
                    )
            # Hybrid CNN cases and 1D CNN cases
            elif re.search(r'CNN', record.model.model_type):
                # Use DeepExplainer when in limited mode
                if re.search(r'Limited', record.model.model_type):
                    feature_score = shap_explainer.get_deep_explain(
                        record.model,
                        trainer_data.all_data.iloc[record.correct_predictions,:]
                    )
                # Use GradientExplainer when in unlimited mode
                elif re.search(r'Unlimited', record.model.model_type):
                    feature_score = shap_explainer.get_gradient_explain(
                        record.model,
                        trainer_data.all_data.iloc[record.correct_predictions,:]
                    )
                else:
                    raise lib.Error('Unknown CNN Type:',record.model.model_type)
            # XGB's GBM cases
            elif record.model.model_type == 'XGB_GBM':
                feature_score = softmax(record.model.clf.feature_importances_)
            # RNN_base model cases
            elif (record.model.model_type == 'RNN' or
                  record.model.model_type == 'LSTM' or
                  record.model.model_type == 'GRU' or
                  record.model.model_type == 'Transformer'):
                feature_score = shap_explainer.get_gradient_explain(
                    record.model,
                    trainer_data.all_data.iloc[record.correct_predictions,:]
                )
            else:
                raise lib.Error('Unknown type: ', record.model.model_type)

            # Update feature_score_sum
            if feature_score_sum is None and feature_score is not None:
                feature_score_sum = pd.array(
                    (feature_score * (1.0 - record.loss)), dtype = float
                )
            elif feature_score is not None:
                feature_score_sum += (feature_score * (1.0 - record.loss))

        # Make feature importnace matrix
        feature_score = pd.DataFrame()
        feature_score.index = bases.columns
        feature_score['importance'] = feature_score_sum
        feature_score = feature_score.sort_values('importance', ascending=False)
        return feature_score

    # clear out Fake GRPs if there is any
    # also subtract DataFrame based on standardized Z score
    def __subtract_feature_score(self, df):
        df['importance'] = df['importance'] - df['importance'][-1]
        remove_list = []
        for ele in df.index:
            if re.search('FAKE', ele) and df.loc[ele]['importance'] != 0.0:
                raise lib.Error('Fake GRP got attention!: ', ele)
                remove_list.append(ele)
        df = df.drop(index = remove_list)
        df = tool.Z_Score_Standardize(df = df, col = 'importance')
        # validation part
        return df

    # Update feature importance matrix with newer matrix
    def add(self, df):
        self.result = self.result.add(df, axis = 0, fill_value = 0).sort_values(
            'importance', ascending = False
        )

    # divide importance value with stabilizing iteration times
    def divide(self, denominator):
        self.result['importance'] = self.result['importance'] / denominator

    # stratify GRPs based on Z score thread
    def stratify(self, z_score_thread, top_grp_amount):
        # change top top_grp_amount to int if value less or equal 1.0
        if top_grp_amount <= 1.0:
            top_grp_amount = int(len(self.result.index) * top_grp_amount)
        for thread in range(len(self.result.index)):
            value = self.result.iloc[thread]['importance']
            if value < z_score_thread or thread == top_grp_amount:
                break
        if thread < top_grp_amount:
            print('Not enough GRP with Z score over thread, extract', thread)
        return self.result[:thread]

    # Save feature importances to given path
    def save(self, path, sep = ','):
        self.result.to_csv(path, sep = sep)



class SHAP_Explainer(object):
    """docstring for SHAP_Explainer."""

    def __init__(self, basement_data = None):
        super(SHAP_Explainer, self).__init__()
        self.basement_data = basement_data

    # Use LinearExplainer
    """ May need to revise """
    def get_linear_explain(self, model, data: pd.DataFrame):
        explainer = shap.LinearExplainer(model, self.basement_data,)
        shap_vals = explainer.shap_values(data)
        return softmax(sum(np.abs(shap_vals).mean(0)))

    # Use KernelExplainer
    def get_kernel_explain(self, model, data: pd.DataFrame):
        # print('Kernel Explainer is too slow! Skipping now!')
        # return None
        explainer = shap.KernelExplainer(model, data = self.basement_data,)
        shap_vals = explainer.shap_values(data)
        return softmax(sum(np.abs(shap_vals).mean(0)))

    # Use GradientExplainer
    def get_gradient_explain(self, model, data: pd.DataFrame):
        # reshape basement data to tensor type
        base = reshape_tensor(self.basement_data.values.tolist())
        explainer = shap.GradientExplainer(model, data = base)
        # Calculate shapley values
        shap_vals = explainer.shap_values(reshape_tensor(data.values.tolist()))
        # Get feature importances based on shapley value
        return softmax(sum(np.abs(shap_vals).mean(0))[0])

    # Use DeepExplainer
    def get_deep_explain(self, model, data: pd.DataFrame):
        # reshape basement data to tensor type
        base = reshape_tensor(self.basement_data.values.tolist())
        explainer = shap.DeepExplainer(model, data = base)
        # Calculate shapley values
        shap_vals = explainer.shap_values(reshape_tensor(data.values.tolist()))
        # Get feature importances based on shapley value
        return softmax(sum(np.abs(shap_vals).mean(0))[0])

    # Use TreeExplainer
    def get_tree_explain(self, model, data: pd.DataFrame):
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation = 'interventional',
            check_additivity = False,
            data = self.basement_data,
        )
        shap_vals = explainer.shap_values(data, check_additivity = False)
        return softmax(sum(np.abs(shap_vals).mean(0)))
