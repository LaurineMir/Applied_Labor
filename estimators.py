import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

def train_random_forest(df,age):
    " trains a random forest model to compare to optimal transport distribution"

    df1=df[df['AGE']==age].reset_index(drop=True)
    df2=df[df['AGE']==age+1].reset_index(drop=True)

    merged=pd.merge(df1,df2,on='nninouv',suffixes=('_'+str(age),'_'+str(age+1)))

    nninouvs = df1['nninouv'].unique()
    X=merged[['SNR_'+str(age),'CS1_'+str(age),'CE_'+str(age),'SX_'+str(age),'cat_'+str(age)]]
    y=merged['SNR_'+str(age+1)]
    X=pd.get_dummies(X,columns=['CS1_'+str(age),'CE_'+str(age),'SX_'+str(age),'cat_'+str(age)],drop_first=True)


    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.9)

    model=RandomForestRegressor(n_estimators=20)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)
    mse=abs(y_test-y_pred)
    print('mae:',mse.mean())


    plt.figure()
    plt.scatter(X_test['SNR_'+str(age)],y_test,color='red')
    plt.scatter(X_test['SNR_'+str(age)], y_pred)
    plt.plot([0,40000],[0,40000],color='grey')
    plt.title('Illustration des Prédiction')
    plt.grid(True)
    plt.xlabel('Salaire initial')
    plt.ylabel('Salaire moyen prédit')
    plt.savefig('c:/Users/Public/Documents/teo&lolo/Plot/Random_forest_predict.png')


    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.title('Illusration des erreurs de prédiction')
    plt.xlabel('Salaire réel')
    plt.ylabel('Salaire prédit')
    plt.savefig('c:/Users/Public/Documents/teo&lolo/Plot/Random_forest_error.png')
    plt.grid(True)
    plt.show()

    return mse



    
    