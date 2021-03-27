def generate_random_missings(df, target, number_of_columns=5):
    for _ in range(number_of_columns):
        col = random.choice(df.columns)
        if col == target:
            pass
        else:
            frac = random.uniform(0, 0.4)
            df.loc[df.sample(frac=frac).index, col] = pd.np.nan
    return df

def cross_validation(model, kf, X_train, y_train):
    list_mse_train = []
    list_mse_val = []
    list_r2_train = []
    list_r2_val = []

    for indices_train, indices_val in kf.split(X_train):
        # iterando sobre os índices de trieno e validação
        X_train_, y_train_ = X_train.loc[indices_train, :], y_train.loc[indices_train]
        X_val_, y_val_ = X_train.loc[indices_val, :], y_train.loc[indices_val]

        # treinando meu modelo no conjunto de treino
        model.fit(X_train_, y_train_)

        # predizendo resultados em treino e validação
        y_pred_train = model.predict(X_train_)
        y_pred_val = model.predict(X_val_)

        # calculando as métricas
        mse_train = mean_squared_error(y_train_, y_pred_train)
        mse_val = mean_squared_error(y_val_, y_pred_val)
        r2_train = r2_score(y_train_, y_pred_train)
        r2_val = r2_score(y_val_, y_pred_val)

        # armazenando resultados
        list_mse_train.append(mse_train)
        list_mse_val.append(mse_val)
        list_r2_train.append(r2_train)
        list_r2_val.append(r2_val)
        
    print('MSE no training: {} +- {}'.format(np.mean(list_mse_train), np.std(list_mse_train)))
    print('MSE no validation: {} +- {}'.format(np.mean(list_mse_val), np.std(list_mse_val)))
    print('r2 no training: {} +- {}'.format(np.mean(list_r2_train), np.std(list_r2_train)))
    print('r2 no validation: {} +- {}'.format(np.mean(list_r2_val), np.std(list_r2_val)))