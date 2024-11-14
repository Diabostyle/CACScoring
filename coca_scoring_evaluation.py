import os
import zipfile
import numpy as np
import pandas as pd
import shutil
import stat
from tqdm import tqdm
import argparse
import math
from math import sqrt
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np
import scipy
import seaborn as sns
import statsmodels.api as sm

################################################################################################
"""
This script analyzes the correlation between Agatston scores and calcium volumes, based on reference and computed scores. 
It classifies cardiovascular risk, calculates correlation coefficients (Pearson, Spearman), and visualizes the results through plots and confusion matrices.

Main functions:
- `label_class_ref` and `label_class_computed`: Classify Agatston scores into different risk categories.
- `extract_data_from_confusion_matrix`: Extracts true and predicted labels from a confusion matrix.
- `bootstrap_cqk`, `bootstrap_pearson`, `bootstrap_spearman`: Estimate the variability of correlation scores using bootstrapping.
- `main`: Manages the primary analysis flow by loading data, calculating statistical correlations, and generating visualizations (scatter plots, Bland-Altman, confusion matrices).

The script also includes statistical analysis and confidence intervals for correlation measurements.
"""
################################################################################################

def label_class_ref(df):
   if df['given_agatston'] < 1:
      return '1'
   if 1 <=df['given_agatston'] < 10:
      return '2'
   if 10 <=df['given_agatston'] < 100:
      return '3'   
   if 100 <=df['given_agatston'] < 400:
      return '4'   
   if df['given_agatston'] >= 400:
      return '5'     

def label_class_computed(df):
   if df['computed_agatston'] < 1:
      return '1'
   if 1 <=df['computed_agatston'] < 10:
      return '2'
   if 10 <=df['computed_agatston'] < 100:
      return '3'   
   if 100 <=df['computed_agatston'] < 400:
      return '4'   
   if df['computed_agatston'] >= 400:
      return '5'     



def extract_data_from_confusion_matrix(confusion_matrix):
    y_true = []
    y_pred = []
    for i, row in enumerate(confusion_matrix):
        for j, qty in enumerate(row):
            y_true.extend([i]*qty)
            y_pred.extend([j]*qty)

    return y_true, y_pred


def bootstrap_cqk(y_true, y_pred):
    num_resamples = 100

    Y = np.array([y_true, y_pred]).T

    weighted_kappas = []
    for i in range(num_resamples):
        Y_resample = np.array(random.choices(Y, k=len(Y)))
        y_true_resample = Y_resample[:, 0]
        y_pred_resample = Y_resample[:, 1]
      #   weighted_kappa = cqk_score(y_true_resample, y_pred_resample)
        weighted_kappa = cohen_kappa_score(y_true_resample,y_pred_resample, labels= ['1','2','3','4','5'], weights = 'linear')
        weighted_kappas.append(weighted_kappa)

    return weighted_kappas

def bootstrap_pearson(y_true, y_pred):
    num_resamples = 100

    Y = np.array([y_true, y_pred]).T

    weighted_r = []
    for i in range(num_resamples):
        Y_resample = np.array(random.choices(Y, k=len(Y)))
        y_true_resample = Y_resample[:, 0]
        y_pred_resample = Y_resample[:, 1]
      #   weighted_kappa = cqk_score(y_true_resample, y_pred_resample)
        r, p_p = scipy.stats.pearsonr(x=y_true_resample, y=y_pred_resample)
        weighted_r.append(r)

    return weighted_r


def bootstrap_spearman(y_true, y_pred):
    num_resamples = 100

    Y = np.array([y_true, y_pred]).T

    weighted_rho = []
    for i in range(num_resamples):
        Y_resample = np.array(random.choices(Y, k=len(Y)))
        y_true_resample = Y_resample[:, 0]
        y_pred_resample = Y_resample[:, 1]
      #   weighted_kappa = cqk_score(y_true_resample, y_pred_resample)
        rho, p_s = scipy.stats.spearmanr(y_true_resample, y_pred_resample) 
        weighted_rho.append(rho)

    return weighted_rho

## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# current working directory in c:
original_working_directory = os.getcwd()

model_dir = os.path.join(current_dir, 'fp_detector')
figure_dir = os.path.join(current_dir, 'Result_Dataset004_COCA2Dv3')                            ## A MODIFIER EN FCT DE FOLDER
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('-filename', type=str, default='coca_score_comparison_0.csv', help="pwd of nii file")

   args = parser.parse_args()
   csv_filename = args.filename

   tabular_dir = os.path.join(current_dir)
   score_comparison_filepath = os.path.join(tabular_dir, csv_filename)
   print ("csv_filename", csv_filename)


   df_score_comparison = pd.read_csv(score_comparison_filepath, sep = ",", na_values=' ')  
   df_score_comparison = df_score_comparison.dropna(subset=['given_agatston'])
   ## Exclude subjects with zero calcium from our evaluation
   df_score_comparison = df_score_comparison[(df_score_comparison["given_agatston"]!=0) | (df_score_comparison["given_agatston"]!=0.0)]
   print ("len(df_score_comparison)", len(df_score_comparison))
   patient_name_list = df_score_comparison["patient_name"] 
   agatston_score_list = df_score_comparison["given_agatston"] 
   computed_agatston_list = df_score_comparison["computed_agatston"] 
   vol_score_list = df_score_comparison["given_vol"]  
   computed_vol_list = df_score_comparison["computed_vol"] 


   csv_filename = csv_filename[:-4]
   model_name = csv_filename.split('_')[-1]
   print ("model_name", model_name)

   df_score_comparison['given_agatston'] = pd.to_numeric(df_score_comparison['given_agatston'], errors='coerce')
   df_score_comparison['computed_agatston'] = pd.to_numeric(df_score_comparison['computed_agatston'], errors='coerce')
   df_score_comparison.dropna(subset=['given_agatston', 'computed_agatston'], inplace=True)

   r, p_p = scipy.stats.pearsonr(x=df_score_comparison["given_agatston"], y=df_score_comparison["computed_agatston"])
   rho, p_s = scipy.stats.spearmanr(df_score_comparison["given_agatston"], df_score_comparison["computed_agatston"]) 
   tau, p_k = scipy.stats.kendalltau(df_score_comparison["given_agatston"], df_score_comparison["computed_agatston"])



   r_object = scipy.stats.pearsonr(x=df_score_comparison["given_agatston"], y=df_score_comparison["computed_agatston"])

   rng = np.random.default_rng()
   method = scipy.stats.BootstrapMethod(n_resamples= 1000, method='BCa', random_state=rng)

   print ("r_object", r_object)
   r_ci = r_object.confidence_interval(confidence_level=0.95)
   print ("r_ci", r_ci)


   rho_object = scipy.stats.spearmanr(df_score_comparison["given_agatston"], df_score_comparison["computed_agatston"]) 
   spearman_r = rho_object.correlation
   count = len(df_score_comparison)

   stderr = 1.0 / math.sqrt(count - 3)
   delta = 1.96 * stderr
   lower = math.tanh(math.atanh(spearman_r) - delta)
   upper = math.tanh(math.atanh(spearman_r) + delta)



   weighted_rs = bootstrap_pearson(df_score_comparison['given_agatston'], df_score_comparison['computed_agatston'])
   r_std =  np.std(weighted_rs)
   print (r_std)
   print ("boot pearsonr lower", r-r_std)
   print ("boot pearsonr upper", r+r_std)


   weighted_rhos = bootstrap_spearman(df_score_comparison['given_agatston'], df_score_comparison['computed_agatston'])
   rho_std =  np.std(weighted_rhos)
   print (rho_std)
   print ("boot spearman lower", rho-rho_std)
   print ("boot spearman upper", rho+rho_std)


   print ("pearsonr", r)   
   print ("spearmanr", rho)


   plt.figure(figsize=(3,3))
   # matplotlib.use('Agg')
   sns.set_theme(style="ticks")
   sns.scatterplot(x="given_agatston", y="computed_agatston", data=df_score_comparison)
   ax = plt.gca() # Get a matplotlib's axes instance
   # plt.title("Agatston score correlation")
   plt.xlabel("Reference Agatston score")
   plt.ylabel("Computed Agatston score")


   plt.text(.55, .32, r"$r$"+"={:.3f}".format(r), transform=ax.transAxes)
   plt.text(.55, .25, r"$\rho$"+"={:.3f}".format(rho), transform=ax.transAxes)
   # plt.text(.6, .25, "Kendall's ={:.3f}".format(tau), transform=ax.transAxes)


   # The following code block adds the correlation line:
   m, b = np.polyfit(df_score_comparison["given_agatston"], df_score_comparison["computed_agatston"], 1)
   X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],10)
   print ("m", m)
   print ("b", b)
   plt.plot(X_plot, m*X_plot + b, 'r-')

   plt.plot(X_plot, X_plot, linestyle = '--', color = 'lime')

   max_value =  max(max(df_score_comparison["given_agatston"]), max(df_score_comparison["computed_agatston"]))
   # plt.ylim(-100, max_value+300)
   # plt.xlim(-100, max_value+300)

   print ("max_value", max_value)

   plt.ylim(-100, max_value+300)
   plt.xlim(-100, max_value+300)


   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)

   plt.savefig(os.path.join(figure_dir, 'agatston_scatter_%s.png' %model_name), bbox_inches='tight', dpi=300)
   plt.savefig(os.path.join(figure_dir, 'agatston_scatter_%s.eps' %model_name), bbox_inches='tight', dpi=300)
   plt.close()

   #create Bland-Altman plot                  
   f, ax = plt.subplots(1, figsize = (4,3))
   sm.graphics.mean_diff_plot(df_score_comparison["given_agatston"], df_score_comparison["computed_agatston"], ax = ax)

   # Labels
   # ax.set_title('Bland-Altman Plot for Agatston score', fontsize = 15)
   ax.set_xlabel('Mean', fontsize = 12)
   ax.set_ylabel('Difference=Reference agatston score' + '\n' + '- Computed agatston score', fontsize = 12)
   # Get axis limits
   left, right = ax.get_xlim()
   bottom, top = ax.get_ylim()
   # Set y-axis limits
   max_y = max(abs(bottom), abs(top))
   # ax.set_ylim(-max_y * 1.1, max_y * 1.1)
   ax.set_ylim(-4000, 4000)
   # Set x-axis limits
   domain = right - left
   # ax.set_xlim(left, left + domain * 1.1)
   ax.set_xlim(left, 7000)

   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)


   #display Bland-Altman plot
   plt.savefig(os.path.join(figure_dir, 'agatston_ba_plot_%s.png' %model_name), bbox_inches='tight', dpi=300)
   plt.savefig(os.path.join(figure_dir, 'agatston_ba_plot_%s.eps' %model_name), bbox_inches='tight', dpi=300)
   plt.close()


   df_score_comparison['ref_class'] = df_score_comparison.apply(label_class_ref, axis=1)
   df_score_comparison['computed_class'] = df_score_comparison.apply(label_class_computed, axis=1)
   print (df_score_comparison)

   for i in range(5):
      i_int = i+1
      df_rank = df_score_comparison[df_score_comparison['ref_class']==str(i_int)]
      print (i_int, len(df_rank))


   k_metric = cohen_kappa_score(df_score_comparison['ref_class'], df_score_comparison['computed_class'], labels= ['1','2','3','4','5'], weights = 'linear')
   print ("k_metric", k_metric)

   cm = confusion_matrix(df_score_comparison['ref_class'], df_score_comparison['computed_class'], labels= ['1','2','3','4','5'])
   ## Plot confusion matrix
   plt.figure(figsize=(5,5))
   plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
   # plt.title('Confusion matrix - Risk categorization' + "\n" r"$\kappa$ =" +"{:.3f}".format(k_metric), size = 15)
   
   plt.colorbar(fraction=0.0458, pad=0.04)
   tick_marks = np.arange(5)
   plt.xticks(tick_marks, ["Very\nlow", "Low", "Moderate", "Moderate\nhigh", "High"], rotation=30, size = 11)
   plt.yticks(tick_marks, ["Very low, \n <1 ", "Low, \n [1,10)", "Moderate, \n [10, 100)", "Mod. high, \n [100, 400)", "High,"+ "\n" r"$\geq$400"], rotation=30, size = 11)
   plt.tight_layout()
   plt.ylabel('Reference agatston score based', size = 12)
   plt.xlabel('Computed agatston score based', size = 12)

   width, height = cm.shape
   for x in range(width):
      for y in range(height):
         plt.annotate(str(cm[x][y]), xy=(y, x), 
         horizontalalignment='center',
         verticalalignment='center', fontsize=15)
   plt.savefig(os.path.join(figure_dir, 'Conf_matrix_agatston_class_%s.png' %model_name), bbox_inches='tight')
   plt.savefig(os.path.join(figure_dir, 'Conf_matrix_agatston_class_%s.eps' %model_name), bbox_inches='tight')
   plt.close()

   ########################
   # Number of classes
   # Sample size
   rows = cm.shape[0]
   cols = cm.shape[1]
   print ("rows", rows)
   print ("cols", cols)
   weights = np.zeros((rows, cols))
   for r in range(rows):
      for c in range(cols):
         # weights[r, c] = float(((r-c)**2)/(rows*cols))
         weights[r, c] = float((abs(r-c))/(rows))
         # weights[r, c] = float(((r-c))/(rows))
         # weights[r, c] = float(((r-c))/(16))
   hist_actual = np.sum(cm, axis=0)
   hist_prediction = np.sum(cm, axis=1)
   expected = np.outer(hist_actual, hist_prediction)
   expected_norm = expected / expected.sum()
   confusion_matrix_norm = cm / cm.sum()
   numerator = 0
   denominator = 0

   for r in range(rows):
      for c in range(cols):
         numerator += weights[r, c] * confusion_matrix_norm[r, c]
         denominator += weights[r, c] * expected_norm[r, c]
   weighted_kappa = (1 - (numerator/denominator))
   print ("weighted_kappa", weighted_kappa)
   total = hist_actual.sum()
   sek = sqrt((numerator * (1 - numerator)) / (total * (1 - denominator) ** 2))
   alpha = 0.95
   margin = (1 - alpha) / 2  # two-tailed test
   x = norm.ppf(1 - margin)
   lower = weighted_kappa - x * sek
   upper = weighted_kappa + x * sek
   # print ("kappa lower", lower)
   # print ("kappa upper", upper)
   #########################
   
   weighted_kappas = bootstrap_cqk(df_score_comparison['ref_class'], df_score_comparison['computed_class'])
   kappa_std =  np.std(weighted_kappas)
   print (kappa_std)
   print ("kappa lower", weighted_kappa-kappa_std)
   print ("kappa upper", weighted_kappa+kappa_std)

   #################
   plt.figure(figsize=(3,3))
   matplotlib.use('Agg')
   sns.set_theme(style="ticks")
   sns.scatterplot(x="given_vol", y="computed_vol", data=df_score_comparison)

   # plt.title("Volume score correlation")
   plt.xlabel("Reference Volume score")
   plt.ylabel("Computed Volume score")

   r, p_p = scipy.stats.pearsonr(x=df_score_comparison["given_vol"], y=df_score_comparison["computed_vol"])
   ax = plt.gca() # Get a matplotlib's axes instance

   rho, p_s = scipy.stats.spearmanr(df_score_comparison["given_vol"], df_score_comparison["computed_vol"]) 
   tau, p_k = scipy.stats.kendalltau(df_score_comparison["given_vol"], df_score_comparison["computed_vol"])


   print ("pearsonr", r)
   print ("pearsonr", p_p)
   print ("spearmanr", rho)
   print ("spearmanr", p_s)


   r_object = scipy.stats.pearsonr(x=df_score_comparison["given_vol"], y=df_score_comparison["computed_vol"])

   rng = np.random.default_rng()
   method = scipy.stats.BootstrapMethod(n_resamples= 1000, method='BCa', random_state=rng)


   rho_object = scipy.stats.spearmanr(df_score_comparison["given_vol"], df_score_comparison["computed_vol"]) 
   spearman_r = rho_object.correlation
   count = len(df_score_comparison)

   stderr = 1.0 / math.sqrt(count - 3)
   delta = 1.96 * stderr
   lower = math.tanh(math.atanh(spearman_r) - delta)
   upper = math.tanh(math.atanh(spearman_r) + delta)


   weighted_rs = bootstrap_pearson(df_score_comparison['given_vol'], df_score_comparison['computed_vol'])
   r_std =  np.std(weighted_rs)
   print (r_std)
   print ("boot pearsonr lower", r-r_std)
   print ("boot pearsonr upper", r+r_std)

   weighted_rhos = bootstrap_spearman(df_score_comparison['given_vol'], df_score_comparison['computed_vol'])
   rho_std =  np.std(weighted_rhos)
   print (rho_std)
   print ("boot spearman lower", rho-rho_std)
   print ("boot spearman upper", rho+rho_std)


   plt.text(.55, .32, r"$r$"+"={:.3f}".format(r), transform=ax.transAxes)
   plt.text(.55, .25, r"$\rho$"+"={:.3f}".format(rho), transform=ax.transAxes)
   # plt.text(.6, .25, "Kendall's ={:.3f}".format(tau), transform=ax.transAxes)


   # The following code block adds the correlation line:
   m, b = np.polyfit(df_score_comparison["given_vol"], df_score_comparison["computed_vol"], 1)
   X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
   plt.plot(X_plot, m*X_plot + b, 'r-')

   plt.plot(X_plot, X_plot, linestyle = '--', color = 'lime')

   max_value =  max(max(df_score_comparison["given_vol"]), max(df_score_comparison["given_vol"]))
   plt.ylim(-100, max_value+800)
   plt.xlim(-100, max_value+800)

   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)


   plt.savefig(os.path.join(figure_dir, 'volume_scatter_%s.png' %model_name), bbox_inches='tight', dpi=300)
   plt.savefig(os.path.join(figure_dir, 'volume_scatter_%s.eps' %model_name), bbox_inches='tight', dpi=300)
   plt.close()


   #create Bland-Altman plot                  
   f, ax = plt.subplots(1, figsize = (4,3))
   sm.graphics.mean_diff_plot(df_score_comparison["given_vol"], df_score_comparison["computed_vol"], ax = ax)

   # Labels
   # ax.set_title('Bland-Altman Plot for Volume score', fontsize = 15)
   ax.set_xlabel('Mean', fontsize = 12)
   ax.set_ylabel('Difference=Reference volume score' + '\n' + '- Computed volume score', fontsize = 12)
   # Get axis limits
   left, right = ax.get_xlim()
   bottom, top = ax.get_ylim()
   # Set y-axis limits
   max_y = max(abs(bottom), abs(top))
   ax.set_ylim(-4000, 4000)
   # Set x-axis limits
   domain = right - left
   ax.set_xlim(left, 7000)

   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)

   #display Bland-Altman plot
   plt.savefig(os.path.join(figure_dir, 'volume_ba_plot_%s.png' %model_name), bbox_inches='tight', dpi=300)
   plt.savefig(os.path.join(figure_dir, 'volume_ba_plot_%s.eps' %model_name), bbox_inches='tight', dpi=300)
   plt.close()

if __name__ == '__main__':
    main()    