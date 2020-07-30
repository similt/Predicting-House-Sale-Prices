#you can download the data file from http://www.stanford.edu/class/stats191/data/ames2000_NAfix.csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

def transform_features(df):
#remove any column with more than 5% missing values
    filter=df.isnull().sum()*100/len(data)
    df=df.drop(filter[filter>5].sort_values().index.tolist(), axis = 1) 
#remove any text column with missing values
    text_mv_counts = df.select_dtypes(include =['object']).isnull().sum().sort_values(ascending=False)
    df.drop(text_mv_counts[text_mv_counts>0].index.tolist(),axis=1,inplace=True)
    
    numc=df.select_dtypes(include =['float64','int64']).isnull().sum()
    numc[numc>0].sort_values()
    dicta={}
## Compute the most common value from each column to replace missing values
    for i in numc[numc>0].sort_values().index.tolist():
        aa=df[i].value_counts().index[0]
        dicta[i]=aa
    for i in dicta: 
        #print(i,dicta[i])
        df[i].fillna(dicta[i],inplace=True)
    text_mv_counts = df.select_dtypes(include =['object']).isnull().sum().sort_values(ascending=False)
    df.drop(text_mv_counts[text_mv_counts>0].index.tolist(),axis=1,inplace=True)
    df['Years Before Sale'] = df['Yr Sold'] - df['Year Built']
    df['Years Since Remod'] = df['Yr Sold'] - df['Year Remod/Add']
    df.drop([2180,1702,2181],axis=0,inplace=True)
# remov unnecessary columns 
    df = df.drop(["Year Built", "Year Remod/Add"], axis = 1)
    df = df.drop(["PID", "Order"], axis=1)
    df = df.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
    return df 
# k=0 : manual splitting of data into  train and test
# k=1 : manually interchange train and test after randomising
# k=2 : use train_test_split to split data into train and test 
# k=3 : Construct k-fold and cross validation is done manually using a loop
# k=4 : Perform k-fold cross validation using k folds 
def select_features(df):
    aa=df[["Gr Liv Area","SalePrice"]]
    return aa
def train_and_test(df,k):
    if k==0:
        train=df[:1460]
        test =df[1460:]
        numerical_train=train.select_dtypes(include =['float64','int64']) 
        numerical_train.drop("SalePrice",axis=1,inplace=True)
        numerical_test=test.select_dtypes(include =['float64','int64']) 
        features =numerical_train.columns
        target = 'SalePrice'
        lr= LinearRegression()
        lr=lr.fit(train[features], train[target])
        a1=lr.coef_
        a0=lr.intercept_
        predictions =lr.predict(test[features])
        mse_test =mean_squared_error(test[target],predictions)
        test_rmse  =np.sqrt(mse_test)
        return test_rmse 
    elif k==1: 
        shuffle_df=df.sample(frac=1, )
        train=df[:1460]
        test =df[1460:]
        numerical_train=train.select_dtypes(include =['float64','int64']) 
        numerical_train.drop("SalePrice",axis=1,inplace=True)
        numerical_test=test.select_dtypes(include =['float64','int64']) 
        features =numerical_train.columns
        target = 'SalePrice'
        lr= LinearRegression()
        
        lr=lr.fit(train[features], train[target])
        predictions_one  =lr.predict(test[features])
        mse_one  =mean_squared_error(test[target],predictions_one)
        rmse_one   =np.sqrt(mse_one)
        
        lr=lr.fit(test[features], test[target])
        predictions_two  =lr.predict(train[features])
        mse_two  =mean_squared_error(train[target],predictions_two)
        rmse_two   =np.sqrt(mse_two)
        avg_rmse=np.mean([rmse_one,rmse_two])
        return avg_rmse
    elif k==2:
        from sklearn.model_selection import train_test_split
        numerical=df.select_dtypes(include =['float64','int64']) 
        target = 'SalePrice'
        features =numerical.columns
        features =features.drop("SalePrice")
        X=numerical[features]
        y=numerical[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        lr= LinearRegression()
        lr=lr.fit(X_train, y_train)
        predict=lr.predict(X_test)
        mse_test =mean_squared_error(y_test,predict)
        test_rmse  =np.sqrt(mse_test)
        return test_rmse 
    elif k==3:
        numerical=df.select_dtypes(include =['float64','int64']) 
        target = 'SalePrice'
        #numerical.drop("SalePrice",axis=1,inplace=True)
        features =numerical.columns
        features =features.drop("SalePrice")
        k=4
        kf = KFold(n_splits=k, shuffle=True)
        lr= LinearRegression()
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse
    
    else: 
        numerical=df.select_dtypes(include =['float64','int64']) 
        target = 'SalePrice'
        #numerical.drop("SalePrice",axis=1,inplace=True)
        features =numerical.columns
        features =features.drop("SalePrice")
        
        kf = KFold(n_splits=k, shuffle=True)
        model =LinearRegression()
        mses = cross_val_score(model, numerical[features], numerical[target], scoring="neg_mean_squared_error", cv=kf)
        rmses = np.sqrt(np.absolute(mses))
        rmse=np.mean(rmses)
        print(rmses)
        return rmse
transform_df = transform_features(data)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df,4)
print(rmse)
