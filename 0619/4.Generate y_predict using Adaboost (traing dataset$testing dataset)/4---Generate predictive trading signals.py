import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util import get_data_raw,train_test_split,\
    lgbm_opt,svm_opt,sgd_opt,gpc_opt,gnb_opt,dtc_opt,ada_opt,gbc_opt,lgbm_opt\
    ,xgb_opt,cat_opt,ridge_opt,mlp_opt,Mlp_opt,knn_opt,gbc_ada_opt_test


#remove 'BTC_RET', 'ETH_RET','rbtc_ret','reth_ret','z_score', 'port_outa_z_score_singel_for_lable','z_score_singel_for_lable'
X_train, y_train = get_data_raw()


#testing data
testing_start_index = '2022-04-01'
testing_end_index = '2023-06-19'


test_dataset = "0617_testing_dataset.csv"
test_dataset = pd.read_csv(test_dataset, parse_dates=[0], index_col=0)
test_dataset = test_dataset.dropna()
# Test period
test_dataset=test_dataset.loc[testing_start_index:testing_end_index, ]

#remove 'BTC_RET', 'ETH_RET','rbtc_ret','reth_ret','z_score', 'port_outa_z_score_singel_for_lable','z_score_singel_for_lable'
X_test = test_dataset.drop(
    columns=['BTC_RET', 'ETH_RET','rbtc_ret','reth_ret',
             'z_score', 'port_outa_z_score_singel_for_lable',
             'z_score_singel_for_lable'])  # , 'port_outa_z_score_singel_for_lable'])

# clf, y_pred=lgbm_opt(X_train, y_train, X_test, y_test)     #0.4850631578947368
# clf, y_pred=svm_opt(X_train, y_train, X_test, y_test)      #0.5013440860215054
# clf, y_pred=sgd_opt(X_train, y_train, X_test, y_test)     #0.5
# clf, y_pred=gpc_opt(X_train, y_train, X_test, y_test)     #0.5068576653459311
# clf, y_pred=gnb_opt(X_train, y_train, X_test, y_test)  # Accuracy: 0.5069306930693069 0.6273503216229589
# clf, y_pred=dtc_opt(X_train, y_train, X_test, y_test)    #0.6247026169706582
# clf, y_pred=ada_opt(X_train, y_train, X_test, y_test)     # Accuracy: 0.7683168316831683 best----------------------0.6837837837837838
# clf, y_pred=gbc_opt(X_train, y_train, X_test, y_test)       #best--------- Accuracy: 0.8198019801980198            0.5522606045212091
# clf, y_pred=lgbm_opt(X_train, y_train, X_test, y_test)      #0.4921721721721722
# clf, y_pred=xgb_opt(X_train, y_train, X_test, y_test)      #0.4318461538461538
# clf, y_pred=cat_opt(X_train, y_train, X_test, y_test)      #0.5492172211350294
# clf, y_pred=ridge_opt(X_train, y_train, X_test, y_test)      # 0.4534368299521689
# clf, y_pred=mlp_opt(X_train, y_train, X_test, y_test)      # 0.5059306198716387
# y_pred=Mlp_opt(X_train, y_train, X_test, y_test)      #  Accuracy: 0.46534653465346537  0.502687164104487
# y_pred=knn_opt(X_train, y_train, X_test, y_test)    # Accuracy: 0.504950495049505


temp1=np.isfinite(X_train).all()
temp2=np.isfinite(X_test).all()
y_pred=gbc_ada_opt_test(X_train, y_train, X_test)  #best--best--best--best--------- Accuracy: 0.8198019801980198



y_pred=pd.DataFrame(y_pred)
y_pred.columns = ['y_pred']
y_pred.index=X_test.index

pd.DataFrame(y_pred).to_csv("../5.Final test/0618_y_prediction.csv", index=True)





# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score, recall_score, f1_score
# p = precision_score(y_test, y_pred, average='micro')
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy:",acc)
# precision_score = precision_score(y_test, y_pred, average='micro')
# print("precision_score:",precision_score)
# r = recall_score(y_test, y_pred, average='micro')
# print("recall_score:",r)
#
# print()
#
