import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ot
import random
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline, LinearNDInterpolator, RBFInterpolator

def remove_upper_quantile(group,col='SNR',q=0.99):
    seuil=group[col].quantile(q)
    return group[group[col] <= seuil]

def get_cleaned_df_from_path(path):
    """entry : path of csv
    output : df with all snr>0 and no nan within all of the rows present in the csv
    """
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['SNR'] = pd.to_numeric(df['SNR'], errors = 'coerce')
    df = df[df['SNR']>0].copy()
    df=df.groupby('AGE', group_keys=False).apply(remove_upper_quantile)
    df['AGE']=df['AGE'].astype(int)
    df['SX'] = df['SX'].apply(lambda x : x.replace("b'","").replace("'",""))        
    df['CS1'] = df['CS1'].apply(lambda x : x.replace("b'","").replace("'",""))        
    df['CE'] = df['CE'].apply(lambda x : x.replace("b'","").replace("'",""))    
    df['cat'] = df['cat'].apply(lambda x : x.replace("b'","").replace("'",""))   
    df['nninouv'] = df['nninouv'].apply(lambda x : x.replace("b'","").replace("'",""))  
    return df

def get_identifiers(df):
    """returns the list of unique identifiers of individual within df"""
    return  df['nninouv'].unique()

def get_annee_unique(df, nninouv):
    """returns (if it exists) the first year where there is only one snr"""
    df_group = df[df['nninouv']==nninouv]
    counts = df_group.groupby("an")["SNR"].transform('count')
    an_unique = df_group.loc[counts==1,"an"]
    if len(an_unique)>0:
        return int(an_unique.iloc[0]), df_group
    else : 
        return None , df_group

def see_marginal(df,column='SNR'):
    """plots the distribution of the attribute column of df"""
    plt.figure()
    if pd.api.types.is_numeric_dtype(df[column]):
        plt.hist(df[column], bins=30, alpha=0.7)
    else :
        counts = df[column].value_counts()
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=45)
    plt.xlabel('Column ' + column)
    plt.ylabel('Counts')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def convert_to_unique_snr(df, verbose=False):
    i=0
    nninouvs = get_identifiers(df)
    if verbose:
        print("Nombres d'identifiants dans df : " + str(len(nninouvs)))
    sample_filtered = pd.DataFrame()
    for nninouv in nninouvs :
        annee_unique, sample_mini = get_annee_unique(df,nninouv)
        i+=1
        if verbose :
             print(i)
        if annee_unique is None :
            df_max=sample_mini.sort_values(['an','SNR'],ascending=[True,False])
            mini_sample_filtered = df_max.drop_duplicates(subset='an',keep="first")
            sample_filtered = pd.concat([sample_filtered,mini_sample_filtered], ignore_index=True)
        if annee_unique is not None :
            sample_filtered = pd.concat([sample_filtered, sample_mini[sample_mini['an']== annee_unique].iloc[0].to_frame().T], ignore_index=True)
            valeur_ref = sample_mini[sample_mini['an']==annee_unique]['SNR'].iloc[0]
            for k in range(1, int(sample_mini['an'].iloc[-1])-annee_unique+1):
                les_salaires = sample_mini[sample_mini['an']== annee_unique + k]['SNR']
                if len(les_salaires)==0:
                    continue
                ecarts = (les_salaires-valeur_ref).abs()
                idx_min = ecarts.idxmin()
                sample_filtered = pd.concat([sample_filtered, sample_mini[sample_mini['an']== annee_unique + k].loc[idx_min].to_frame().T], ignore_index=True)
                valeur_ref = les_salaires.loc[idx_min]

            valeur_ref = sample_mini[sample_mini['an']==annee_unique]['SNR'].iloc[0]
            for k in range(1, annee_unique-int(sample_mini['an'].iloc[0])+1):

                les_salaires = sample_mini[sample_mini['an']== annee_unique - k]['SNR']
                if len(les_salaires)==0:
                    continue
                ecarts = (les_salaires-valeur_ref).abs()
                idx_min = ecarts.idxmin()
                sample_filtered = pd.concat([sample_filtered, sample_mini[sample_mini['an']== annee_unique - k].loc[idx_min].to_frame().T], ignore_index=True)
                valeur_ref = les_salaires.loc[idx_min]
    return sample_filtered


def convert_snr(df):
    snr_sum=df.groupby(['an','nninouv'], as_index=False)['SNR'].sum()
    df_unique=df.drop_duplicates(subset=['an','nninouv'],keep='first').copy()
    df_unique=df_unique.drop(columns=['SNR']).merge(snr_sum,on=['an','nninouv'],how='left')
    return df_unique


def get_present_age(df,age):
    return not df[df['AGE']==age].empty


def get_sampled_df_age(df,age,nb_samples,nb_joint,verbose=False):
    """returns a local_df with two different values of age with 
    at least nb_samples for age, and at least nb_joint identifiers 
    common between the samples"""
    df_age = df[((df['AGE']==age) | (df['AGE']==age+1))]
    rows_list = []
    nninouvs =get_identifiers(df_age)
    random.shuffle(nninouvs)
    i = 0 #parcours nninouv
    k=0 #gets to nb_joint
    n_age = 0
    n_next_age=0 #gets to nb_samples-nb_joint
    while (k<nb_joint)| (n_age < nb_samples) | (n_next_age < nb_samples):
        if verbose :
            print(f"k:{k}, n_age:{n_age}, n_age_next:{n_next_age}")
        nninouv = nninouvs[i]
        df_nninouv = df_age[df_age['nninouv']==nninouv]
        present_age = get_present_age(df_nninouv,age)
        present_next_age = get_present_age(df_nninouv,age+1)
        is_joint = present_age & present_next_age
        if is_joint :
            rows_list.append(df_nninouv)
            k+=1
            n_age += 1
            n_next_age += 1
        elif not is_joint and n_age < nb_samples and present_age :
            rows_list.append(df_nninouv)
            n_age += 1
        elif not is_joint and n_next_age < nb_samples and present_next_age :
            rows_list.append(df_nninouv)
            n_next_age += 1
        i+=1
    return pd.concat(rows_list, ignore_index=True)


def get_subset_joint(df,age):
    ids = df['nninouv'][df['AGE']==age]
    ids_next = df['nninouv'][df['AGE']==age+1]
    df_commun = df[((df['AGE']==age) & (df['nninouv'].isin(ids_next)))|((df['AGE']==age+1) & (df['nninouv'].isin(ids)))]
    return df_commun

def real_joint(sample_filtered,age,n=5,column='SNR'):
    sample_filtered_30 = sample_filtered[sample_filtered['AGE']==age]
    sample_filtered_31 = sample_filtered[sample_filtered['AGE']==age+1]

    pd.qcut(sample_filtered_30['SNR'], q=n, labels=False)
    sample_filtered_30['salaire_cat']=pd.qcut(sample_filtered_30[column], q=n, labels=False)
    bins = pd.qcut(sample_filtered_30[column], q=n)
    cat_30 = sample_filtered_30[['nninouv','salaire_cat']]

    sample_filtered_31= sample_filtered_31.merge(cat_30, on = 'nninouv', how='left')

    n= sample_filtered_31['salaire_cat'].nunique()
    colors = plt.cm.viridis(np.linspace(0,1,n))
    plt.figure()
    for i in range(n):
        plt.bar(i,1,color=colors[i])
        plt.text(i,1.05, str(i),ha="center",va='bottom', fontsize =12)

    plt.figure()   
    for cat in range(n) :
        subset =sample_filtered_31[sample_filtered_31['salaire_cat']==cat]['SNR']
        print(subset)
        plt.hist(subset, bins=30, alpha=1, label='Categorie {cat}', color = colors[cat])

    plt.title(f'Histogramme de {column}')
    plt.show()


def cout(data1, data2, alpha=100, beta=100, gamma=100, delta=100):
    # data1 et data2 sont des dictionnaires
    cout_salaire = abs(data1['SNR']-data2['SNR'])**2
    cout_cs = np.where(data1['CS1']== data2['CS1'],1,0)
    cout_cat = np.where(data1['cat']== data2['cat'],1,0)
    cout_genre = np.where(data1['SX']== data2['SX'],1,0)
    cout_partiel = np.where(data1['CE']== data2['CE'],1,0)
    return cout_salaire/10000000+alpha*cout_cs + beta*cout_cat + gamma*cout_genre + delta*cout_partiel

def transport_annee(df,age, alpha=100, beta=100, gamma=100, delta=100,verbose=False,compare_distrib=False,category=False):
    df1=df[df['AGE']==age].reset_index(drop=True)
    df2=df[df['AGE']==age+1].reset_index(drop=True)

    n1,n2=len(df1),len(df2)
    if verbose:
        print("début du calcul du cout")
    C=np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            C[i,j]=cout(df1.loc[i],df2.loc[j],alpha, beta, gamma, delta)

    a = np.ones(n1)/n1
    b = np.ones(n2)/n2

    if verbose:
        print("début du OT")
    G=ot.emd(a,b,C)
    print(G)
    if verbose:
        print("fin du OT")
    transport_indiv = G @ df2['SNR'].values
    transport_indiv = transport_indiv / G.sum(axis=1)

    if verbose:
        print("début du graphe")
    plt.figure()
    if compare_distrib:
        salaires_T2 = []
        for nninouv in df1['nninouv'].values:
             salaires_T2.append(df2[df2['nninouv']==nninouv]['SNR'].values[0])
        plt.scatter(df1['SNR'].values,salaires_T2,color='red')
    if verbose :
        if not category :
            plt.scatter(df1['SNR'].values, transport_indiv)
        if category :
            plt.scatter(df1['SNR'].values, transport_indiv,c=df1['CE'].apply(lambda x: 0 if x=='C' else 1).values, cmap='viridis')
            plt.colorbar(label='CE')  
    
        plt.plot([0,40000],[0,40000],color='grey')
    
        plt.title('Illustration du plan de transport')
        plt.grid(True)
        plt.xlabel('Salaire initial')
        plt.ylabel('Salaire moyen prédit')
        if compare_distrib:
            if verbose:
                print("début du graphe compare")
            plt.figure()
            plt.scatter(salaires_T2, transport_indiv)  
            plt.title('Illusration des erreurs de prédiction')
            plt.xlabel('Salaire réel')
            plt.ylabel('Salaire prédit')
            plt.grid(True)
            print('mae'+str(abs(salaires_T2-transport_indiv).mean()))
        plt.show()
    
    return G

def compare_mobility(df1,df2,G,age):
    transport_indiv = G @ df2['SNR'].values
    transport_indiv = transport_indiv / G.sum(axis=1)
    salaires_T2 = []
    for nninouv in df1['nninouv'].values:
        salaires_T2.append(df2[df2['nnivouv']==nninouv]['SNR'][0])
    
def get_tuples(df,age):
    grouped = df.groupby('nninouv')
    tuple_list=[]
    for ident,group in grouped:
        age_data=group[group['AGE']==age]
        age_next=group[group['AGE']==age+1]
        if not age_data.empty and not age_next.empty :
            snr = age_data['SNR'].values[0] 
            snr_NEXT = age_next['SNR'].values[0]
            ce = age_data['CE'].values[0]  
            cat = age_data['CAT'].values[0] 
            cs1 = age_data['CS1'].values[0]  
            sx = age_data['SX'].values[0]

            tuple = (snr,cs1,cat,sx,ce,snr_NEXT)
            tuple_list.append(tuple)
    return tuple_list      

def get_index(nninouv,joint1):
    return joint1.index[joint1['nninouv']==nninouv].tolist()[0]

def get_expected_salary(joint1,joint2,nninouv,P):
    index = get_index(nninouv,joint1)
    return np.dot(P[index], joint2['SNR'].values)/np.sum(P, axis=1)[index]


def best_alpha(df,age, n_iterations, verbose=False, init=6):
    joint1=df[df['AGE']==age].reset_index(drop=True)
    joint2=df[df['AGE']==age+1].reset_index(drop=True)
    n1,n2=len(joint1),len(joint2)

    p1 = np.ones(n1)/n1
    p2 = np.ones(n2)/n2

    best_test_loss=float('inf')

    samples_df = get_subset_joint(df,age)
    ident = get_identifiers(samples_df)
    best = [10**init,10**init,10**init,10**init]
    best_list=[best]
    for n in range(n_iterations):
        if verbose:
            print(f"iteration {n}")
        to_test = [best]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        to_test.append([best[0]*np.power(10.0,-1+2*i),best[1]*np.power(10.0,-1+2*j),best[2]*np.power(10.0,-1+2*k),best[3]*np.power(10.0,-1+2*l)])
        for i in range(3):
            random_choice = []
            powers = np.random.randint(1,6,size=4)
            for power in powers :
                random_choice.append(np.power(10.0,power))
            to_test.append(random_choice)

        for alpha,beta, gamma, delta in to_test :
            if verbose:
                print("alpha : "+ str(alpha))
                print("beta : "+ str(beta))
                print("gamma : "+ str(gamma))
                print("delta : "+ str(delta))
            C=np.zeros((n1,n2))
            for i in range(n1):
                for j in range(n2):
                    C[i,j]=cout(joint1.loc[i],joint2.loc[j], alpha=alpha, beta = beta, gamma = gamma, delta = delta)

            P = ot.emd(p1,p2,C)
            if verbose :
                print("plan de transport calculé")
            loss_test=0
            for nninouv in ident:
                expected = get_expected_salary(joint1,joint2,nninouv,P)
                truth = joint2[joint2['nninouv']==nninouv]['SNR'].values[0]
                loss_test += np.sqrt(abs(expected-truth))
            loss_test=(loss_test)**2/len(samples_df)


            if loss_test < best_test_loss:
                if verbose :
                    print(f'meilleure erreur obtenue pour {alpha}, {beta}, {gamma}, {delta}, erreur : {loss_test}')
                best_test_loss=loss_test
                best=[alpha,beta,gamma,delta]
                best_list.append(best)
                best_test_loss = loss_test


    return best,best_list


def recup_values(line):
    ''' found the values of age, alpha, beta, gamma of the fiel params.txt'''
    parts = line.strip().split(',')
    values={}
    for part in parts :
        key,val = part.split(':')
        values[key.strip()]=float(val.strip())
    return values


def get_next_salary(revenu, genre,cat_socio,G,marg1,marg2,seuil_log=0.5):
    subset = marg1[
        (marg1['SX'] == genre) &
        (marg1['CS1'] == cat_socio) &
        (np.abs(np.log(marg1['SNR'])-np.log(revenu)) <= seuil_log)
    ]

    if subset.empty :
        print('erreur')
        return np.nan
    
    expected_salaries = subset['nninouv'].apply(lambda nn: get_expected_salary(marg1, marg2, nn, G))
    
    return expected_salaries.mean()


def write_str(G_dict):
    for age in range(20,61):
        for fea in ['SX','CS1']:
            G_dict[f"df_{age}_0"][fea]=G_dict[f"df_{age}_0"][fea].apply(lambda x : str(x))
            G_dict[f"df_{age}_0"][fea]=G_dict[f"df_{age}_1"][fea].apply(lambda x : str(x))
    return G_dict


def trajctoires_seuil(G_dict,genre,cs1,seuil,revenu_inf=500,revenu_sup=3000,nb_revenu = 10,plot=False):
    revenus_initiaux=np.linspace(revenu_inf,revenu_sup,nb_revenu)
    trajectoires={}
    for revenu in revenus_initiaux:
        trajectoires[f'{revenu}'] = [revenu]
        for age in range(20,61) :
            G=G_dict[age]
            marg1 = G_dict[f"df_{age}_0"]
            marg2= G_dict[f"df_{age}_1"]
            expected = get_next_salary(trajectoires[f'{revenu}'][-1],genre,cs1,G,marg1=marg1,marg2=marg2, seuil_log=seuil)

            trajectoires[f'{revenu}'].append(expected)

    if plot :
            plt.figure(figsize=(8,5))
            for revenu_init, salaire in trajectoires.items():
                annees = list(range(len(salaire)))
                plt.plot(annees, salaire, marker='o',label=f'Rev_init={revenu_init[:5]}')
            plt.ylabel("Salaire")
            plt.xlabel("Annees")
            plt.grid(True)
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            plt.show()
    return trajectoires



def smooth_salary(s_values,s_plus_values,age_values,alphas=np.logspace(-5,2,50),degree=3,n_knots_s=6,n_knots_a=6,verbose=False):
    X=np.vstack((s_values,age_values)).T
    y=np.array(s_plus_values)

    spline =SplineTransformer(degree=degree,n_knots =n_knots_a, include_bias=True)
    kf = KFold(n_splits=5,shuffle=True)
    mean_mse=[]

    for alpha in alphas:
        mse_list=[]
        for train_idx,test_idx in kf.split(X):
            X_train=spline.fit_transform(X[train_idx])
            X_test=spline.transform(X[test_idx])
            y_train=y[train_idx]
            y_test=y[test_idx]

            model=Ridge(alpha=alpha)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            mse_list.append(mean_squared_error(y_test,y_pred))
        mean_mse.append(np.mean(mse_list))
    best_alpha = alphas[np.argmin(mean_mse)]

    if verbose:
        plt.figure(figsize=(6,4))
        plt.plot(np.log10(alphas),mean_mse,marker='o')
        plt.axvline(np.log10(best_alpha),color='red',linestyle='--',label=f'alpha optimal ={best_alpha:.2e}]')
        plt.xlabel('log alpha')
        plt.ylabel('mse croisé')
        plt.title('Régularisation')
        plt.legend()
        plt.grid(True)
        plt.show()                    
    spline = SplineTransformer(degree=degree,n_knots =n_knots_a, include_bias=True)
    model= make_pipeline(spline,Ridge(alpha=best_alpha))
    model.fit(X,y)

    def f(s_grid,a_grid):
        sa=np.vstack((s_grid,a_grid)).T
        return model.predict(sa)
    
    return f,best_alpha

def get_s_age_tuples(G_dict, genre="none", cs="none") : 
    s_values=[]
    age_values=[]
    s_plus_values=[]
    for age in range(20,61):
        df_age=G_dict[f"df_{age}_0"]
        if genre != "none":
            df_age = df_age[df_age['SX']==genre]
        if cs != "none":
            df_age = df_age[df_age["CS1"]==cs]
        for idx,line in  df_age.iterrows():
          
            age_values.append(age)
            nninouv = line['nninouv']
            s_plus_values.append(get_expected_salary(df_age,G_dict[f"df_{age}_1"],nninouv,G_dict[age]))
            s_values.append(line['SNR'])
    return s_values,s_plus_values,age_values

def spline_salary(s_values,s_plus_values,age_values,s=0):
    X=np.vstack((s_values,age_values)).T
    y=np.array(s_plus_values)
    spline = RBFInterpolator(X, y, kernel='thin_plate_spline',smoothing=s)            
    # spline = RBFInterpolator(s_values, age_values, s_plus_values,s=s)            

    def f(s_grid,a_grid):
        sa=np.vstack((s_grid,a_grid)).T
        s_vals=spline(sa)
        return s_vals.reshape(s_grid.shape)
    
    return f

def unsmooth_salary(s_values,s_plus_values,age_values,s=0):
    X=np.vstack((s_values,age_values)).T
    y=np.array(s_plus_values)
    inter = LinearNDInterpolator(X,y)            

    def f(s_grid,a_grid):
        sa=np.vstack((s_grid,a_grid)).T
        s_vals=inter(sa)
        return s_vals.reshape(s_grid.shape)
    
    return f
def mean_trajectories(G_dict,genre,cs):
    s_values,s_plus_values,age_values = get_s_age_tuples(G_dict, genre=genre, cs=cs)
    f=unsmooth_salary(s_values,s_plus_values,age_values, s=0)

    s_grid_1D =np.linspace(np.min(s_values),np.max(s_values),50)

    s_mesh, a_mesh = np.meshgrid(s_grid_1D,age_values)
    s_flat = s_mesh.ravel()
    a_flat = a_mesh.ravel()
    U = np.ones_like(s_flat)
    V = f(s_flat,a_flat)

    plt.figure(figsize=(12,6))
    plt.quiver(a_flat,s_flat,U, V-s_flat, V, cmap='viridis', angles="xy", scale_units = 'xy', scale=1)
    plt.xlabel("age")
    plt.ylabel("salaire")
    plt.title(f"champ de vecteurs de salaire en fonction de l'age pour les {dict_cs[cs]}")
    plt.grid(True)
    plt.xticks([i for i in range(20,61)])
    plt.tight_layout()
    plt.savefig(f"c:/Users/Public/Documents/teo&lolo/trajectories_{cs}.png")
    plt.show()