
TRAIN_FILES = ['../data/arabic/', # 0
               '../data/CK/', # 1
               '../data/character/', # 2
               '../data/Action3D/', # 3
               '../data/Activity/', # 4
               '../data/arabic_voice/', # 5
               '../data/JapaneseVowels/', # 6

               # New benchmark datasets
               '../data/AREM/', # 7
               '../data/gesture_phase/', # 8
               '../data/HT_Sensor/', # 9
               '../data/MovementAAL/', # 10
               '../data/HAR/', # 11
               '../data/occupancy_detect/', # 12
               '../data/eeg/',  # 13
               '../data/ozone/', # 14
               '../data/daily_sport/',  # 15
               '../data/eeg2/',  # 16
               '../data/MHEALTH/',  # 17
               '../data/EEG_Comp2_data1a/',  # 18
               '../data/EEG_Comp2_data1b/',  # 19
               '../data/EEG_Comp2_data3/',  # 20
               '../data/EEG_Comp2_data4/',  # 21
               '../data/EEG_Comp3_data2/',  # 22
               '../data/EEG_Comp3_data1/',  # 23
               '../data/uwave/',  # 24
               '../data/opportunity/',  # 25
               '../data/pamap2/',  # 26
               '../data/WEASEL_MUSE_DATASETS/ArabicDigits/',  # 27
               '../data/WEASEL_MUSE_DATASETS/AUSLAN/',  # 28
               '../data/WEASEL_MUSE_DATASETS/CharacterTrajectories/',  # 29
               '../data/WEASEL_MUSE_DATASETS/CMUsubject16/',  # 30
               '../data/WEASEL_MUSE_DATASETS/ECG/',  # 31
               '../data/WEASEL_MUSE_DATASETS/JapaneseVowels/',  # 32
               '../data/WEASEL_MUSE_DATASETS/KickvsPunch/',  # 33
               '../data/WEASEL_MUSE_DATASETS/Libras/',  # 34
               '../data/WEASEL_MUSE_DATASETS/NetFlow/',  # 35
               '../data/WEASEL_MUSE_DATASETS/PEMS/',  # 36
               '../data/WEASEL_MUSE_DATASETS/UWave/',  # 37
               '../data/WEASEL_MUSE_DATASETS/Wafer/',  # 38
               '../data/WEASEL_MUSE_DATASETS/WalkvsRun/',  # 39
               '../data/WEASEL_MUSE_DATASETS/digitshape_random/',  # 40
               '../data/WEASEL_MUSE_DATASETS/lp1/',  # 41
               '../data/WEASEL_MUSE_DATASETS/lp2/',  # 42
               '../data/WEASEL_MUSE_DATASETS/lp3/',  # 43
               '../data/WEASEL_MUSE_DATASETS/lp4/',  # 44
               '../data/WEASEL_MUSE_DATASETS/lp5/',  # 45
               '../data/WEASEL_MUSE_DATASETS/pendigits/',  # 46
               '../data/WEASEL_MUSE_DATASETS/shapes_random/',  # 47
               ]

TEST_FILES = ['../data/arabic/', # 0
              '../data/CK/', # 1
              '../data/character/', # 2
              '../data/Action3D/', # 3
              '../data/Activity/', # 4
              '../data/arabic_voice/', # 5
              '../data/JapaneseVowels/', # 6

              # New benchmark datasets
              '../data/AREM/', # 7
              '../data/gesture_phase/', # 8
              '../data/HT_Sensor/',  # 9
              '../data/MovementAAL/',  # 10
              '../data/HAR/',  # 11
              '../data/occupancy_detect/',  # 12
              '../data/eeg/', # 13
              '../data/ozone/',  # 14
              '../data/daily_sport/',  # 15
              '../data/eeg2/',  # 16
              '../data/MHEALTH/',  # 17
              '../data/EEG_Comp2_data1a/',  # 18
              '../data/EEG_Comp2_data1b/',  # 19
              '../data/EEG_Comp2_data3/',  # 20
              '../data/EEG_Comp2_data4/',  # 21
              '../data/EEG_Comp3_data2/',  # 22
              '../data/EEG_Comp3_data1/',  # 23
              '../data/uwave/',  # 24
              '../data/opportunity/',  # 25
              '../data/pamap2/',  # 26
              '../data/WEASEL_MUSE_DATASETS/ArabicDigits/',  # 27
              '../data/WEASEL_MUSE_DATASETS/AUSLAN/',  # 28
              '../data/WEASEL_MUSE_DATASETS/CharacterTrajectories/',  # 29
              '../data/WEASEL_MUSE_DATASETS/CMUsubject16/',  # 30
              '../data/WEASEL_MUSE_DATASETS/ECG/',  # 31
              '../data/WEASEL_MUSE_DATASETS/JapaneseVowels/',  # 32
              '../data/WEASEL_MUSE_DATASETS/KickvsPunch/',  # 33
              '../data/WEASEL_MUSE_DATASETS/Libras/',  # 34
              '../data/WEASEL_MUSE_DATASETS/NetFlow/',  # 35
              '../data/WEASEL_MUSE_DATASETS/PEMS/',  # 36
              '../data/WEASEL_MUSE_DATASETS/UWave/',  # 37
              '../data/WEASEL_MUSE_DATASETS/Wafer/',  # 38
              '../data/WEASEL_MUSE_DATASETS/WalkvsRun/',  # 39
              '../data/WEASEL_MUSE_DATASETS/digitshape_random/',  # 40
              '../data/WEASEL_MUSE_DATASETS/lp1/',  # 41
              '../data/WEASEL_MUSE_DATASETS/lp2/',  # 42
              '../data/WEASEL_MUSE_DATASETS/lp3/',  # 43
              '../data/WEASEL_MUSE_DATASETS/lp4/',  # 44
              '../data/WEASEL_MUSE_DATASETS/lp5/',  # 45
              '../data/WEASEL_MUSE_DATASETS/pendigits/',  # 46
              '../data/WEASEL_MUSE_DATASETS/shapes_random/',  # 47

              ]

MAX_NB_VARIABLES = [13, # 0
                    136, # 1
                    30, # 2
                    570, # 3
                    570, # 4
                    39, # 5
                    12, # 6

                    # New benchmark datasets
                    7, # 7
                    18, # 8
                    11, # 9
                    4, # 10
                    9, # 11
                    5, # 12
                    13, # 13
                    72, # 14
                    45, # 15
                    64, # 16
                    23, # 17
                    6, # 18
                    7, # 19
                    3, # 20
                    28, # 21
                    64, # 22
                    64, # 23
                    3, # 24
                    77, # 25
                    52, # 26
                    13, # 27
                    22, # 28
                    3, # 29
                    62, # 30
                    2, # 31
                    12, # 32
                    62, # 33
                    2, # 34
                    4, # 35
                    963, # 36
                    3, # 37
                    6, # 38
                    62,# 39
                    2, # 40
                    6, # 41
                    6, # 42
                    6, # 43
                    6, # 44
                    6, # 45
                    2, # 46
                    2, # 47

                    ]

MAX_TIMESTEPS_LIST = [93, # 0
                      71, # 1
                      173, # 2
                      100, # 3
                      337, # 4
                      91, # 5
                      26, # 6

                      # New benchmark datasets
                      480, # 7
                      214, # 8
                      5396, # 9
                      119, # 10
                      128, # 11
                      3758, # 12
                      117, # 13
                      291, # 14
                      125, # 15
                      256, # 16
                      42701, # 17
                      896, # 18
                      1152, # 19
                      1152, # 20
                      500,# 21
                      7794, # 22
                      3000, # 23
                      315, # 24
                      24, # 25
                      34, # 26
                      93, # 27
                      96, # 28
                      205, # 29
                      534, # 30
                      147, # 31
                      26, # 32
                      761, # 33
                      45, # 34
                      994, # 35
                      144, # 36
                      315, # 37
                      198, # 38
                      1918, # 39
                      97, # 40
                      15, # 41
                      15, # 42
                      15, # 43
                      15, # 44
                      15, # 45
                      8, # 46
                      97, # 47

                      ]


NB_CLASSES_LIST = [10, # 0
                   7, # 1
                   20, # 2
                   20, # 3
                   16, # 4
                   88, # 5
                   9, # 6

                   # New benchmark datasets
                   7, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   6, # 11
                   2, # 12
                   2, # 13
                   2, # 14
                   19, # 15
                   2, # 16
                   13, # 17
                   2, # 18
                   2, # 19
                   2, # 20
                   2, # 21
                   29, # 22
                   2, # 23
                   8, # 24
                   18, # 25
                   12, # 26
                   10, # 27
                   95, # 28
                   20, # 29
                   2, # 30
                   2, # 31
                   9, # 32
                   2, # 33
                   15, # 34
                   2, # 35
                   7, # 36
                   8, # 37
                   2, # 38
                   2, # 39
                   4, # 40
                   4, # 41
                   5, # 42
                   4, # 43
                   3, # 44
                   5, # 45
                   10, # 46
                   3, # 47

                   ]

if __name__ == '__main__':
  print('max_nb_vars\tmax_timesteps\tnum_classes\tdata_dir') 
  from mlstmfcn.utils import constants as c
  for max_nb_vars, max_timesteps, num_classes, data_dir in zip(c.MAX_NB_VARIABLES, c.MAX_TIMESTEPS_LIST, c.NB_CLASSES_LIST, c.TRAIN_FILES): 
    if num_classes == 2: #binary
    # if max_nb_vars == 2: # simple features
    # if max_timesteps < 25: # small time axis
      print('{}\t\t{}\t\t{}\t\t{}'.format(max_nb_vars, max_timesteps, num_classes, data_dir)) 
'''
max_nb_vars max_timesteps num_classes data_dir
13    93    10    ../data/arabic/
136   71    7   ../data/CK/
30    173   20    ../data/character/
570   100   20    ../data/Action3D/
570   337   16    ../data/Activity/
39    91    88    ../data/arabic_voice/
12    26    9   ../data/JapaneseVowels/
7   480   7   ../data/AREM/
18    214   5   ../data/gesture_phase/
11    5396    3   ../data/HT_Sensor/
4   119   2   ../data/MovementAAL/
9   128   6   ../data/HAR/
5   3758    2   ../data/occupancy_detect/
13    117   2   ../data/eeg/
72    291   2   ../data/ozone/
45    125   19    ../data/daily_sport/
64    256   2   ../data/eeg2/
23    42701   13    ../data/MHEALTH/
6   896   2   ../data/EEG_Comp2_data1a/
7   1152    2   ../data/EEG_Comp2_data1b/
3   1152    2   ../data/EEG_Comp2_data3/
28    500   2   ../data/EEG_Comp2_data4/
64    7794    29    ../data/EEG_Comp3_data2/
64    3000    2   ../data/EEG_Comp3_data1/
3   315   8   ../data/uwave/
77    24    18    ../data/opportunity/
52    34    12    ../data/pamap2/
13    93    10    ../data/WEASEL_MUSE_DATASETS/ArabicDigits/
22    96    95    ../data/WEASEL_MUSE_DATASETS/AUSLAN/
3   205   20    ../data/WEASEL_MUSE_DATASETS/CharacterTrajectories/
62    534   2   ../data/WEASEL_MUSE_DATASETS/CMUsubject16/
2   147   2   ../data/WEASEL_MUSE_DATASETS/ECG/
12    26    9   ../data/WEASEL_MUSE_DATASETS/JapaneseVowels/
62    761   2   ../data/WEASEL_MUSE_DATASETS/KickvsPunch/
2   45    15    ../data/WEASEL_MUSE_DATASETS/Libras/
4   994   2   ../data/WEASEL_MUSE_DATASETS/NetFlow/
963   144   7   ../data/WEASEL_MUSE_DATASETS/PEMS/
3   315   8   ../data/WEASEL_MUSE_DATASETS/UWave/
6   198   2   ../data/WEASEL_MUSE_DATASETS/Wafer/
62    1918    2   ../data/WEASEL_MUSE_DATASETS/WalkvsRun/
2   97    4   ../data/WEASEL_MUSE_DATASETS/digitshape_random/
6   15    4   ../data/WEASEL_MUSE_DATASETS/lp1/
6   15    5   ../data/WEASEL_MUSE_DATASETS/lp2/
6   15    4   ../data/WEASEL_MUSE_DATASETS/lp3/
6   15    3   ../data/WEASEL_MUSE_DATASETS/lp4/
6   15    5   ../data/WEASEL_MUSE_DATASETS/lp5/
2   8   10    ../data/WEASEL_MUSE_DATASETS/pendigits/
2   97    3   ../data/WEASEL_MUSE_DATASETS/shapes_random/
'''