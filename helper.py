def numerical(option,col,df_missing):
    if option == "Mean":
         df_missing[col].fillna(df_missing[col].mean(), inplace=True)
    elif option == "Median":
        df_missing[col].fillna(df_missing[col].median(), inplace=True)
    elif option == "Mode":
        df_missing[col].fillna(df_missing[col].mode()[0], inplace=True)
    elif option == "Interquartile Range (IQR)":
        Q1 = df_missing[col].quantile(0.25)
        Q3 = df_missing[col].quantile(0.75)
        IQR = Q3 - Q1
        df_missing[col].fillna(df_missing[col].median(), inplace=True)

    return df_missing

