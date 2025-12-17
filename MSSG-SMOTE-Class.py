import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from pymof import MOF

from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from collections import defaultdict

import smote_variants as sv
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE

from collections import Counter
from sklearn.decomposition import PCA


class MSSG :
    def __init__(self):
        pass

    def mof_scores(self, data):
        '''
        Function to calculate MOF scores from input Data
        '''
        model = MOF()
        model.fit(data)
        scores = model.decision_scores_
        return scores

    def add_class(self, data):
        '''
        Calculate quartile of MOF scores and add them as new columns  
        '''
        scores = self.mof_scores(data)
        quartiles = np.quantile(scores, [0.25, 0.5, 0.75])
        quartile_classes = np.digitize(scores, bins=quartiles, right=True) + 1 
        return quartile_classes
    
    def Directional(self , data , X_maj , X_min , all_sample):
        '''
        data        : instance in that we focus in this case 
        X_maj       : Majority instance features 
        X_min       : All of minority instance features 
        all_sample  : number of Synthetic instance --> number of Synthetic ins + data
        '''
        print(f'data : {len(data)}')

        #find majority NearestNeighbors instance from data
        nbrs = NearestNeighbors(n_neighbors=5).fit(X_maj) # we set n_neighbor = 5 because the maximum tainted value
        distances, indices = nbrs.kneighbors(data)

        num_sample = (all_sample)/len(data) #number of synthetic instance for each data instance 
        num_nearest_min = round(num_sample)
        
        print(f'num_nearest_min : {num_nearest_min}')
        print(f'n_samples_fit : {len(X_min)}')

        if num_nearest_min > len(X_min) :
            num_nearest_min = len(X_min)

        #find minority num_sample NearestNeighbors instance from data 
        #we want the number of NearestNeighbors = number the instance that we want to synthesize 
        nbrs_class1 = NearestNeighbors(n_neighbors = num_nearest_min ).fit(X_min) 
        distances_select , indices_select = nbrs_class1.kneighbors(data)
        
        #print(indices_select)

        synthetic_samples = []

        number_loop = round(all_sample / (len(data) + len(data)*num_nearest_min))
        
        if number_loop < 1 : 
            number_loop = 1
        print(f'number_loop : {number_loop}')
        for l in range(number_loop):
            for n in range(num_nearest_min) :
                for i, ins in enumerate(data):
                    #print(f'data[{i}] --> nearest[{n}]')
                    nearest_min_points = X_min[indices_select[i]]
                    nearest_min_ins = nearest_min_points[n] 

                    norm_diff = np.linalg.norm(nearest_min_ins - ins)

                    if norm_diff == 0:
                        alpha = 1
                    else:
                        ball_radius = distances[i][0] / norm_diff
                        alpha = np.random.uniform(0, ball_radius)  # Random proportion along the line  

                    synthetic_sample = ins + alpha * (nearest_min_ins - ins)
                    synthetic_samples.append(synthetic_sample)

        print(f'len(synthetic_samples : {len(synthetic_samples)}')
        return synthetic_samples
    
    def sample(self , X,y):
        X_min = X[y == 1] 
        y_min = y[y == 1]
        
        X_maj = X[y == 0]
        y_maj = y[y == 0]

        synthetic_sample = X_maj
        synthetic_target = y_maj

        quartile_all = self.add_class(X)
        quartile_minority = self.add_class(X_min)

        quartile_df = pd.DataFrame({
            'feature': X[y == 1].tolist(),
            'target': y[y == 1],
            'quartile_all': quartile_all[y == 1],
            'quartile_minority': quartile_minority
        })


        drop_df1 = quartile_df[(quartile_df['quartile_all'].isin([1,2]) & quartile_df['quartile_minority'].isin([1,2]))]
        quartile_df = quartile_df[~(quartile_df['quartile_all'].isin([1,2]) & quartile_df['quartile_minority'].isin([1,2]))]

        drop_df2 = quartile_df[(quartile_df['quartile_all'].isin([4]) & quartile_df['quartile_minority'].isin([4]))]
        quartile_df = quartile_df[~(quartile_df['quartile_all'].isin([4]) & quartile_df['quartile_minority'].isin([4]))]

        print('+++++++++++++++++++++++++++++++++++++++++')
        print(f'drop instance : {len(drop_df1) + len(drop_df2)}')

        synthetic_sample = np.vstack((synthetic_sample , np.array(drop_df1['feature'].tolist())))
        synthetic_target = np.hstack((synthetic_target , np.array(drop_df1['target'].tolist())))

        synthetic_sample = np.vstack((synthetic_sample , np.array(drop_df2['feature'].tolist())))
        synthetic_target = np.hstack((synthetic_target , np.array(drop_df2['target'].tolist())))

        print(f'len(synthetic array) : {len(synthetic_sample)}')
        print(f'remaining synthetic instance : {(len(X[y==0])*2) - len(synthetic_sample)}')
        print('+++++++++++++++++++++++++++++++++++++++++')

        #?==================================================================================
        X_d = len(drop_df1) + len(drop_df2)
        X_expec = len(X) - (X_d)
        print(f'X_expec : {X_expec}')
        
        r = (len(X_min)) / (len(X))
        final_synthesize = round(((1-r)) * (len(X)))
        final_synthesize = final_synthesize - X_d
        number_synthesize = final_synthesize/9
        
        print(f'X = {len(X)} , X_maj = {len(X_maj)} , X_min = {len(X_min)}')
        print(f'r = {r} and x = {number_synthesize} | ({number_synthesize *9}) where X - X_min = {(len(X) - len(X_min))}')
        #?==================================================================================

        combinations = quartile_df[['quartile_minority' , 'quartile_all']].values.tolist()
        combination_counts = dict(Counter(map(tuple, combinations)))

        filtered_combinations = {key: value for key, value in combination_counts.items()}
        print(len(filtered_combinations))
        print(filtered_combinations)

        print(f'number that expectation majority = {len(X_maj)} minority = {len(X_min)} : {len(X_maj) - len(X_min)}')
        print(f'number of synthetic instance : {(len(X_maj) - len(X_min))/len(filtered_combinations)}')

        ratio_list = [number_synthesize , number_synthesize , 4*number_synthesize , 3*number_synthesize]
        print(f'Ratio : {ratio_list}')

        for combi in filtered_combinations : 
            if combi[0] < combi[1] :
                select_df = quartile_df[(quartile_df['quartile_minority'] == combi[0]) & (quartile_df['quartile_all'] == combi[1])]
                
                ratio_syn = ratio_list[combi[0] - 1] * (len(select_df) / len(quartile_df[quartile_df['quartile_minority'].isin([combi[0]])]))
                print(f'ratio syn in case [{combi}] : {ratio_syn}')

                synthetic = []
                select_df = select_df.drop_duplicates(subset=['feature'])
                select_df_feature = np.array(select_df['feature'].tolist())

                new_sample = self.Directional(select_df_feature, X_maj , X_min , round(ratio_syn) )
                print(f'synthetic instance = {len(new_sample)}')

                synthetic.append(new_sample)

                if len(synthetic[0]) == 0 : 
                    continue
                else :
                    synthetic_array = np.vstack(synthetic)
                    synthetic_sample = np.vstack((synthetic_sample , synthetic_array))
                    synthetic_target = np.hstack((synthetic_target , [1]*len(synthetic_array)))   
                    print(f'len(synthetic array) : {len(synthetic_sample)}')            

            else : 
                select_df = quartile_df[(quartile_df['quartile_minority'] == combi[0]) & (quartile_df['quartile_all'] == combi[1] )]

                ratio_syn = ratio_list[combi[0] - 1] * (len(select_df) / len(quartile_df[quartile_df['quartile_minority'].isin([combi[0]])]))
                print(f'ratio syn in case [{combi}] : {ratio_syn}')

                # when number of instance <= 1 skip this case 
                if len(select_df) <= 1 : 
                    continue

                # if number of instance < 6 can't use SMOTE with parameter Neighbors = 5 but we can use DSS method
                elif len(select_df) <= 10 :
                    select_df = quartile_df[(quartile_df['quartile_minority'] == combi[0]) & (quartile_df['quartile_all'] == combi[1] )]
                    synthetic = []
                    select_df = select_df.drop_duplicates(subset=['feature'])
                    select_df_feature = np.array(select_df['feature'].tolist())

                    new_sample = self.Directional(select_df_feature, X_maj , X_min , round(ratio_syn))
                    print(f'synthetic instance = {len(new_sample)}')
                    synthetic.append(new_sample)
                    if len(synthetic[0]) == 0 : 
                        continue
                    else :
                        synthetic_array = np.vstack(synthetic)
                        synthetic_sample = np.vstack((synthetic_sample , synthetic_array))
                        synthetic_target = np.hstack((synthetic_target , [1]*len(synthetic_array)))   
                        print(f'len(synthetic array) : {len(synthetic_sample)}')     

                else :
                    ratio_SMOTE = ((ratio_syn)) / len(X_maj)
                    
                    print(ratio_SMOTE)

                    select_df_feature = np.vstack((X_maj,np.array(select_df['feature'].tolist())))
                    select_df_target = np.hstack((y_maj ,np.array(select_df['target'].tolist())))
                    smote = SMOTE(sampling_strategy= ratio_SMOTE ,random_state=42)
                    X_border , y_border = smote.fit_resample(select_df_feature,select_df_target)
                    print(f'synthetic instance | SMOTE = {len(X_border[y_border == 1])}')
                    synthetic_sample = np.vstack((synthetic_sample , X_border[y_border == 1]))
                    synthetic_target = np.hstack((synthetic_target , y_border[y_border == 1]))
                    print(f'len(synthetic array) : {len(synthetic_sample)}')
            
        return synthetic_sample , synthetic_target