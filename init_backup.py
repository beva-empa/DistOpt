    #
    # count = 0
    # for rt in range(num_hubs):
    #     for st in range(num_hubs):
    #         if st != rt:
    #             for t in range(num_opt_var):
    #                 p_avg_in[t][count] = 0.0
    #                 z_list_P_in[t][count] = 0.0
    #                 h_avg_in[t][count] = 0.0
    #                 z_list_H_in[t][count] = 0.0
    #
    #             count += 1

    # p_avg_in = np.array([[115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0], [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0],
    #                      [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0], [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0],
    #                      [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0], [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0],
    #                      [4.873816601, 30.64327485, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    #                      [189.9481191, 25.32701764, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    #                      [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0]])  # ,
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [61.96044351, 9.591975694, 0.0, 0.0, 0.0, 0.0],
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    # [172.7137372, 26.43274854, 0.0, 0.0, 0.0, 0.0], [172.7137372, 26.43274854, 0.0, 0.0, 0.0, 0.0],
    # [172.7137372, 26.43274854, 0.0, 0.0, 0.0, 0.0], [132.3976608, 18.01169591, 0.0, 0.0, 0.0, 0.0]])

    # z_list_P_in = np.array(
    #     [[115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0], [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0],
    #      [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0], [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0],
    #      [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0], [115.5555556, 15.90643275, 0.0, 0.0, 0.0, 0.0],
    #      [4.873816601, 30.64327485, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    #      [189.9481191, 25.32701764, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    #      [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0]])  # ,
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [61.96044351, 9.591975694, 0.0, 0.0, 0.0, 0.0],
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    # [0, 0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0.0, 0.0, 0.0, 0.0],
    # [172.7137372, 26.43274854, 0.0, 0.0, 0.0, 0.0], [172.7137372, 26.43274854, 0.0, 0.0, 0.0, 0.0],
    # [172.7137372, 26.43274854, 0.0, 0.0, 0.0, 0.0], [132.3976608, 18.01169591, 0.0, 0.0, 0.0, 0.0]])

    # z_list_H_in = np.array([[0.0, 0.0, 16.63571547, 0.0, 4.4, 0.0], [0.0, 0.0, 17.98548633, 0.0, 7.3, 0.0],
    #                         [0.0, 0.0, 21.98548633, 0.0, 5.5, 0.0], [0.0, 0.0, 18.98548633, 0.0, 9.3, 0.0],
    #                         [0.0, 0.0, 6.985486335, 0.0, 7.3, 0.0], [0.0, 0.0, 6, 0.0, 4.4, 0.0],
    #                         [0.0, 0.0, 16.97108878, 0.0, 8.6, 0.0], [0.0, 0.0, 0, 0.0, 0, 0.0],
    #                         [4.706871885, 0.0, 0, 0.0, 0, 0.0], [12.08100304, 0.0, 0, 0.0, 0, 0.0],
    #                         [2.925447481, 0.0, 0, 0.0, 0, 0.0], [0.0, 0.0, 0, 0.0, 0.141219186, 0.0]])  # ,
    # [0.0, 0.0, 15.99709727, 0.0, 9.3, 0.0], [0.0, 0.0, 32.99709727, 0.0, 12.9, 0.0],
    # [0.0, 0.0, 32.54573074, 0.0, 13.6, 0.0], [0.0, 0.0, 29.43685286, 0.0, 11.3, 0.0],
    # [0.0, 0.0, 29.54573074, 0.0, 10, 0.0], [0.0, 0.0, 26.98548633, 0.0, 5.7, 0.0],
    # [0.0, 0.0, 22.98548633, 0.0, 10, 0.0], [0.0, 0.0, 13.98548633, 0.0, 9.3, 0.0],
    # [0.0, 0.0, 11.98548633, 0.0, 3.7, 0.0], [0.0, 0.0, 11.98548633, 0.0, 9.3, 0.0],
    # [0.0, 0.0, 0, 0.0, 9.3, 0.0], [0.0, 0.0, 3.979088779, 0.0, 5.7, 0.0]])

    # h_avg_in = np.array([[0.0, 0.0, 16.63571547, 0.0, 4.4, 0.0], [0.0, 0.0, 17.98548633, 0.0, 7.3, 0.0],
    #                      [0.0, 0.0, 21.98548633, 0.0, 5.5, 0.0], [0.0, 0.0, 18.98548633, 0.0, 9.3, 0.0],
    #                      [0.0, 0.0, 6.985486335, 0.0, 7.3, 0.0], [0.0, 0.0, 6, 0.0, 4.4, 0.0],
    #                      [0.0, 0.0, 16.97108878, 0.0, 8.6, 0.0], [0.0, 0.0, 0, 0.0, 0, 0.0],
    #                      [4.706871885, 0.0, 0, 0.0, 0, 0.0], [12.08100304, 0.0, 0, 0.0, 0, 0.0],
    #                      [2.925447481, 0.0, 0, 0.0, 0, 0.0], [0.0, 0.0, 0, 0.0, 0.141219186, 0.0]])  # ,
    # [0.0, 0.0, 15.99709727, 0.0, 9.3, 0.0], [0.0, 0.0, 32.99709727, 0.0, 12.9, 0.0],
    # [0.0, 0.0, 32.54573074, 0.0, 13.6, 0.0], [0.0, 0.0, 29.43685286, 0.0, 11.3, 0.0],
    # [0.0, 0.0, 29.54573074, 0.0, 10, 0.0], [0.0, 0.0, 26.98548633, 0.0, 5.7, 0.0],
    # [0.0, 0.0, 22.98548633, 0.0, 10, 0.0], [0.0, 0.0, 13.98548633, 0.0, 9.3, 0.0],
    # [0.0, 0.0, 11.98548633, 0.0, 3.7, 0.0], [0.0, 0.0, 11.98548633, 0.0, 9.3, 0.0],
    # [0.0, 0.0, 0, 0.0, 9.3, 0.0], [0.0, 0.0, 3.979088779, 0.0, 5.7, 0.0]])

    # lagran_P_in = np.array([[[-0.169499908, -0.146483747, 0.130143847, 0.0, 0.130144949, 0.0],
    #                          [0.169499908, 0.0, -0.130143847, -0.219496213, 0.0, 0.195272431],
    #                          [0.0, 0.146483747, 0.0, 0.219496213, -0.130144949, -0.195272431]],
    #                         [[-0.169260616, -0.14491539, 0.130046352, 0.0, 0.130047514, 0.0],
    #                          [0.169260616, 0.0, -0.130046352, -0.219494771, 0.0, 0.193873331],
    #                          [0.0, 0.14491539, 0.0, 0.219494771, -0.130047514, -0.193873331]],
    #                         [[-0.169500038, -0.145282937, 0.130149317, 0.0, 0.130150546, 0.0],
    #                          [0.169500038, 0.0, -0.130149317, -0.219494532, 0.0, 0.194008066],
    #                          [0.0, 0.145282937, 0.0, 0.219494532, -0.130150546, -0.194008066]],
    #                         [[-0.169499914, -0.145528215, 0.124741758, 0.0, 0.124741146, 0.0],
    #                          [0.169499914, 0.0, -0.124741758, -0.219493615, 0.0, 0.19426642],
    #                          [0.0, 0.145528215, 0.0, 0.219493615, -0.124741146, -0.19426642]],
    #                         [[-0.169499957, -0.145525618, 0.127643526, 0.0, 0.127644889, 0.0],
    #                          [0.169499957, 0.0, -0.127643526, -0.219492335, 0.0, 0.194263761],
    #                          [0.0, 0.145525618, 0.0, 0.219492335, -0.127644889, -0.194263761]],
    #                         [[-0.169500041, -0.145497247, 0.130140988, 0.0, 0.130142498, 0.0],
    #                          [0.169500041, 0.0, -0.130140988, -0.219491602, 0.0, 0.194233686],
    #                          [0.0, 0.145497247, 0.0, 0.219491602, -0.130142498, -0.194233686]],
    #                         [[-0.20104835, -0.20104831, 0.21404324, 0.0, 0.214048705, 0.0],
    #                          [0.20104835, 0.0, -0.21404324, -0.219488477, 0.0, 0.219510843],
    #                          [0.0, 0.201048312, 0.0, 0.219488477, -0.214048705, -0.219510843]],
    #                         [[-0.268271472, -0.26825489, 0.268229762, 0.0, 0.268246602, 0.0],
    #                          [0.268271472, 0.0, -0.268229762, -0.268233045, 0.0, 0.268266761],
    #                          [0.0, 0.26825489, 0.0, 0.268233045, -0.268246602, -0.268266761]],
    #                         [[-0.193249933, -0.157345183, 0.131257975, 0.0, 0.13125879, 0.0],
    #                          [0.193249933, 0.0, -0.131257975, -0.268230027, 0.0, 0.230455005],
    #                          [0.0, 0.157345183, 0.0, 0.268230027, -0.13125879, -0.230455005]],
    #                         [[-0.121783648, -0.121752525, 0.122213454, 0.0, 0.122239877, 0.0],
    #                          [0.121783648, 0.0, -0.122213454, -0.121968235, 0.0, 0.122025514],
    #                          [0.0, 0.121752525, 0.0, 0.121968235, -0.122239877, -0.122025514]],
    #                         [[-0.121530312, -0.121485119, 0.122468965, 0.0, 0.122517194, 0.0],
    #                          [0.121530312, 0.0, -0.122468965, -0.121952063, 0.0, 0.12204619],
    #                          [0.0, 0.121485119, 0.0, 0.121952063, -0.122517194, -0.12204619]],
    #                         [[-0.121623237, -0.121581996, 0.122441989, 0.0, 0.122489071, 0.0],
    #                          [0.121623237, 0.0, -0.122441989, -0.121953066, 0.0, 0.122045029],
    #                          [0.0, 0.121581996, 0.0, 0.121953066, -0.122489071, -0.122045029]]])  # ,
    # [[-0.119838137, -0.11981501, 0.122487769, 0.0, 0.122536788, 0.0],
    #  [0.119838137, 0.0, -0.122487769, -0.121949601, 0.0, 0.122047412],
    #  [0.0, 0.11981501, 0.0, 0.121949601, -0.122536788, -0.122047412]],
    # [[-0.121656683, -0.121639702, 0.12249758, 0.0, 0.122545918, 0.0],
    #  [0.121656683, 0.0, -0.12249758	, -0.121951217, 0.0, 0.12204648],
    #  [0.0, 0.121639702, 0.0, 0.121951217, -0.122545918, -0.12204648]],
    # [[-0.115926898, -0.115888045, 0.122513789, 0.0, 0.122562538, 0.0],
    #  [0.115926898, 0.0, -0.122513789, -0.121949246, 0.0, 0.122047372],
    #  [0.0, 0.115888045, 0.0, 0.121949246, -0.122562538, -0.122047372]],
    # [[-0.144828169, -0.139394872, 0.124243728, 0.0, 0.124242929, 0.0],
    #  [0.144828169, 0.0, -0.124243728, -0.138626735, 0.0, 0.152162025],
    #  [0.0, 0.139394872, 0.0, 0.138626735, -0.124242929, -0.152162025]],
    # [[-0.26746946	, -0.267434676, 0.268124368, 0.0, 0.268124778, 0.0],
    #  [0.26746946, 0.0, -0.268124368, -0.268231258, 0.0, 0.268270544],
    #  [0.0, 0.267434676, 0.0, 0.268231258, -0.268124778, -0.268270544]],
    # [[-0.267599548, -0.267584587, 0.269016423, 0.0, 0.269036523, 0.0],
    #  [0.267599548, 0.0, -0.269016423, -0.268229570, 0.0, 0.268270113],
    #  [0.0, 0.267584587, 0.0, 0.26822957, -0.269036523, -0.268270113]],
    # [[-0.267632553, -0.267617483, 0.278149624, 0.0, 0.278174361, 0.0],
    #  [0.267632553, 0.0, -0.278149624, -0.268228793, 0.0, 0.268269984],
    #  [0.0, 0.267617483, 0.0, 0.268228793, -0.278174361, -0.268269984]],
    # [[-0.267608743, -0.267592834, 0.26838006, 0.0, 0.268381809, 0.0],
    #  [0.267608743, 0.0, -0.26838006, -0.268227914, 0.0, 0.268270935],
    #  [0.0, 0.267592834, 0.0, 0.268227914, -0.268381809, -0.268270935]],
    # [[-0.250156667, -0.239250463, 0.268336285, 0.0, 0.249215132, 0.0],
    #  [0.250156667, 0.0, -0.268336285, -0.268187073, 0.0, 0.256769868],
    #  [0.0, 0.239250463, 0.0, 0.268187073, -0.249215132, -0.256769868]],
    # [[-0.201647855, -0.193300717, 0.220455197, 0.0, 0.195085664, 0.0],
    #  [0.201647855, 0.0, -0.220455197, -0.220252944, 0.0, 0.210713313],
    #  [0.0, 0.193300717, 0.0, 0.220252944, -0.195085664, -0.210713313]],
    # [[-0.204897566, -0.19893967, 0.221532142, 0.0, 0.203915691, 0.0],
    #  [0.204897566, 0.0, -0.221532142, -0.220255273, 0.0, 0.213228564],
    #  [0.0, 0.19893967, 0.0, 0.220255273, -0.203915691, -0.213228564]],
    # [[-0.121879564, -0.12188033, 0.122521445, 0.0, 0.126643814, 0.0],
    #  [0.121879564, 0.0, -0.122521445, -0.121995941, 0.0, 0.121999319],
    #  [0.0, 0.12188033, 0.0, 0.121995941, -0.126643814, -0.121999319]]])

    # lagran_H_in = np.array([[[-0.106568998, -0.109392079, 0.116624266, 0.0, 0.120700945, 0.0],
    #                          [0.106568998, 0.0, -0.116624266, -0.115472323, 0.0, 0.10656948],
    #                          [0.0, 0.109392079, 0.0, 0.115472323, -0.120700945, -0.10656948]],
    #                         [[-0.107571612, -0.109497613, 0.11011943, 0.0, 0.116046877, 0.0],
    #                          [0.107571612, 0.0, -0.11011943, -0.106410356, 0.0, 0.107572699],
    #                          [0.0, 0.109497613, 0.0, 0.106410356, -0.116046877, -0.107572699]],
    #                         [[-0.105357933, -0.109118497, 0.108746146, 0.0, 0.116106614, 0.0],
    #                          [0.105357933, 0.0, -0.108746146, -0.101754095, 0.0, 0.105358967],
    #                          [0.0, 0.109118497, 0.0, 0.101754095, -0.116106614, -0.105358967]],
    #                         [[-0.105976347, -0.110067666, 0.110899934, 0.0, 0.115262429, 0.0],
    #                          [0.105976347, 0.0, -0.110899934, -0.102283681, 0.0, 0.10597764],
    #                          [0.0, 0.110067666, 0.0, 0.102283681, -0.115262429, -0.10597764]],
    #                         [[-0.114891066, -0.119028352, 0.111591926, 0.0, 0.114612432, 0.0],
    #                          [0.114891066, 0.0, -0.111591926, -0.1044768, 0.0, 0.11087527],
    #                          [0.0, 0.119028352, 0.0, 0.1044768, -0.114612432, -0.11087527]],
    #                         [[-0.120976808, -0.123464411, 0.113387589, 0.0, 0.1153328, 0.0],
    #                          [0.120976808, 0.0, -0.113387589, -0.105404684, 0.0, 0.112006702],
    #                          [0.0, 0.123464411, 0.0, 0.105404684, -0.1153328, -0.112006702]],
    #                         [[-0.1083275, -0.118117104, 0.116585455, 0.0, 0.116461299, 0.0],
    #                          [0.1083275, 0.0, -0.116585455, -0.105325165, 0.0, 0.108333523],
    #                          [0.0, 0.118117104, 0.0, 0.105325165, -0.116461299, -0.108333523]],
    #                         [[-0.141649409, -0.127336411, 0.143849617, 0.0, 0.119720209, 0.0],
    #                          [0.141649409, 0.0, -0.143849617, -0.127354133, 0.0, 0.143526915],
    #                          [0.0, 0.127336411, 0.0, 0.127354133, -0.119720209, -0.143526915]],
    #                         [[-0.142546524, -0.130678998, 0.144937435, 0.0, 0.12019303, 0.0],
    #                          [0.142546524, 0.0, -0.144937435, -0.133036462, 0.0, 0.144324065],
    #                          [0.0, 0.130678998, 0.0, 0.133036462, -0.12019303, -0.144324065]],
    #                         [[-0.100725469, -0.12186732, 0.114280443, 0.0, 0.114432059, 0.0],
    #                          [0.100725469, 0.0, -0.114280443, -0.103171093, 0.0, 0.10161399],
    #                          [0.0, 0.12186732, 0.0, 0.103171093, -0.114432059, -0.10161399]],
    #                         [[-0.145591457, -0.138778493, 0.163758945, 0.0, 0.138580486, 0.0],
    #                          [0.145591457, 0.0, -0.163758945, -0.141314978, 0.0, 0.154167732],
    #                          [0.0, 0.138778493, 0.0, 0.141314978, -0.138580486, -0.154167732]],
    #                         [[-0.130111603, -0.130702098, 0.137381477, 0.0, 0.112778886, 0.0],
    #                          [0.130111603, 0.0, -0.137381477, -0.137469273, 0.0, 0.114743912],
    #                          [0.0, 0.130702098, 0.0, 0.137469273, -0.112778886, -0.114743912]]])  # ,
    # [[-0.101504245,-0.0939433	, 0.106248398, 0.0, 0.109083126, 0.0],
    #  [0.101504245, 0.0, -0.106248398, -0.097754986, 0.0, 0.092251731],
    #  [0.0, 0.0939433, 0.0, 0.097754986, -0.109083126, -0.092251731]],
    # [[-0.127429588,-0.127853133, 0.099007726, 0.0, 0.109421214, 0.0],
    #  [0.127429588, 0.0, -0.099007726, -0.089589857, 0.0, 0.105796172],
    #  [0.0, 0.127853133, 0.0, 0.089589857, -0.109421214, -0.105796172]],
    # [[-0.12765946,-0.1281442	, 0.108945466, 0.0, 0.11601039, 0.0],
    #  [0.12765946, 0.0, -0.108945466, -0.097769792, 0.0, 0.098865785],
    #  [0.0, 0.1281442, 0.0, 0.097769792, -0.11601039	, -0.098865785]],
    # [[-0.124057991,-0.123885303, 0.098227479, 0.0, 0.105079246, 0.0],
    #  [0.124057991, 0.0, -0.098227479, -0.090552507, 0.0, 0.094335205],
    #  [0.0, 0.123885303, 0.0, 0.090552507, -0.105079246, -0.094335205]],
    # [[-0.105864691,-0.090723558, 0.13225326, 0.0, 0.123780113, 0.0],
    #  [0.105864691, 0.0, -0.13225326	, -0.089526008, 0.0, 0.085796558],
    #  [0.0, 0.090723558, 0.0, 0.089526008, -0.123780113, -0.085796558]],
    # [[-0.111693242,-0.110907767, 0.108044213, 0.0, 0.107186241, 0.0],
    #  [0.111693242, 0.0, -0.108044213, -0.100651503, 0.0, 0.098771878],
    #  [0.0, 0.110907767, 0.0, 0.100651503, -0.107186241, -0.098771878]],
    # [[-0.10713415,-0.119537247, 0.108874618, 0.0, 0.111244107, 0.0],
    #  [0.10713415, 0.0, -0.108874618, -0.105168708, 0.0, 0.098378736],
    #  [0.0, 0.119537247, 0.0, 0.105168708, -0.111244107, -0.098378736]],
    # [[-0.122642394,-0.124463641, 0.111566289, 0.0, 0.114468288, 0.0],
    #  [0.122642394, 0.0, -0.111566289, -0.105647468, 0.0, 0.111780334],
    #  [0.0, 0.124463641, 0.0, 0.105647468, -0.114468288, -0.111780334]],
    # [[-0.100515323,-0.095739543, 0.110404002, 0.0, 0.110466516, 0.0],
    #  [0.100515323, 0.0, -0.110404002, -0.096124809, 0.0, 0.100579095],
    #  [0.0, 0.095739543, 0.0, 0.096124809, -0.110466516, -0.100579095]],
    # [[-0.099434742,-0.103522098, 0.111595059, 0.0, 0.11506775, 0.0],
    #  [0.099434742, 0.0, -0.111595059, -0.103060078, 0.0, 0.099406254],
    #  [0.0, 0.103522098, 0.0, 0.103060078, -0.11506775	, -0.099406254]],
    # [[-0.113066826,-0.117612311, 0.112907219, 0.0, 0.117817239, 0.0],
    #  [0.113066826, 0.0, -0.112907219, -0.107454862, 0.0, 0.111691134],
    #  [0.0, 0.117612311, 0.0, 0.107454862, -0.117817239, -0.111691134]],
    # [[-0.115150286,-0.122680291, 0.113361008, 0.0, 0.116613916, 0.0],
    #  [0.115150286, 0.0, -0.113361008, -0.105963457, 0.0, 0.115622061],
    #  [0.0, 0.122680291, 0.0, 0.105963457, -0.116613916, -0.115622061]]])

    # for hub in range(num_hubs):
    #     count = 0
    #     for rt in range(num_hubs):
    #         for st in range(num_hubs):
    #             if st != rt:
    #                 if rt == hub:
    #                     for t in range(num_opt_var):
    #                         lagran_P_in[t][hub][count] = -0.16
    #                         lagran_H_in[t][hub][count] = -0.10
    #                 elif st == hub:
    #                     for t in range(num_opt_var):
    #                         lagran_P_in[t][hub][count] = 0.16
    #                         lagran_H_in[t][hub][count] = 0.10
    #                 else:
    #                     for t in range(num_opt_var):
    #                         lagran_P_in[t][hub][count] = 0.0
    #                         lagran_H_in[t][hub][count] = 0.0
    #                 count += 1