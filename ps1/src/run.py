from LogisticRegression import main as p01b
from GDA import main as gda
from Q2_posonly import main as posonly
from poisson import main as poisson
from WeightedLinear import main as weight
from tau import main as tau

'''p01b(train_path='../cs229-ps-2018-master/ps1/data/ds1_train.csv',
     eval_path='../cs229-ps-2018-master/ps1/data/ds1_valid.csv',
     pred_path='../p01b_pred_1.txt')

gda(train_path='../cs229-ps-2018-master/ps1/data/ds1_train.csv',
     eval_path='../cs229-ps-2018-master/ps1/data/ds1_valid.csv',
     pred_path='../p01e_pred_1.txt')

posonly(train_path='../cs229-ps-2018-master/ps1/data/ds3_train.csv',
     valid_path='../cs229-ps-2018-master/ps1/data/ds3_valid.csv',
     test_path='../cs229-ps-2018-master/ps1/data/ds3_test.csv',
     pred_path='../p02_pred_X.txt')

poisson(lr = .0000001, train_path='../cs229-ps-2018-master/ps1/data/ds4_train.csv',
     eval_path='../cs229-ps-2018-master/ps1/data/ds4_valid.csv',
     pred_path='../p03_pred_1.txt')

weight(.5,train_path='../cs229-ps-2018-master/ps1/data/ds5_train.csv',
     eval_path='../cs229-ps-2018-master/ps1/data/ds5_valid.csv')'''

tau(tau_values=[5e-2, 1e-1, 5e-1, 1, 10], train_path='../cs229-ps-2018-master/ps1/data/ds5_train.csv',
     valid_path='../cs229-ps-2018-master/ps1/data/ds5_valid.csv',
     test_path='../cs229-ps-2018-master/ps1/data/ds5_test.csv',
     pred_path='../p05_pred_2.txt')
