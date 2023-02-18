# predict_income

![image](https://user-images.githubusercontent.com/97363174/219844178-9333224e-b66d-4085-90e6-0db43eed3b20.png)

![image](https://user-images.githubusercontent.com/97363174/219844193-084a7e28-a034-402d-bb1f-7308d566ddd6.png)

![image](https://user-images.githubusercontent.com/97363174/219844203-0617c1b0-25a5-4a11-bf4a-1f812bcd515a.png)

Confusion Matrix on test set:            Predicted 0  Predicted 1
Actual 0         4640          305
Actual 1          886          682
Accuracy Score on test set:  0.8171349608475357
                  Variable      Coef
1                      age  2.268350
2             capital-gain  1.624467
3             capital-loss  1.119745
6                 sex_Male  1.028060
4           hours-per-week  0.937521
5           hours-per-week  0.665296
23           education-num  0.159857
7            education-num  0.145070
37   education_Prof-school  0.026613
22  education_Some-college  0.016227
38  education_Some-college  0.000489
35       education_Masters  0.000000
32     education_Bachelors  0.000000
31     education_Assoc-voc  0.000000
29           education_9th  0.000000
28       education_7th-8th  0.000000
27       education_5th-6th  0.000000
26       education_1st-4th  0.000000
25          education_12th  0.000000
36     education_Preschool  0.000000
24          education_11th  0.000000
33     education_Doctorate  0.000000
19       education_Masters  0.000000
21   education_Prof-school  0.000000
20     education_Preschool  0.000000
18       education_HS-grad  0.000000
17     education_Doctorate  0.000000
16     education_Bachelors  0.000000
15     education_Assoc-voc  0.000000
13           education_9th  0.000000
12       education_7th-8th  0.000000
11       education_5th-6th  0.000000
10       education_1st-4th  0.000000
9           education_12th  0.000000
8           education_11th  0.000000
14    education_Assoc-acdm -0.029265
30    education_Assoc-acdm -0.091534
34       education_HS-grad -0.117442
0                Intercept -7.453282

AUC: 0.8447306210148366
