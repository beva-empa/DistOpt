
            z_imp_pwr = []
            z_imp_ht = []

            for hub in range(num_hubs):

                z_imp = []
                z_ex = []

                for idx in range(num_hubs-1):
                    z_imp.append(z_local(num_opt_var))
                    z_ex.append(z_local(num_opt_var))

                cost1 = 0
                temp1 = [0.0] * num_opt_var
                temp2 = [0.0] * num_opt_var

                for idx in range(len(z_ex)):
                    for t in range(num_opt_var):
                        temp1[t] += transfer_pw_exp[hub][idx][t] - z_ex[idx].P_strt[t]
                        temp2[t] += transfer_ht_exp[hub][idx][t] - z_ex[idx].Q_strt[t]
                        cost1 += (z_ex[idx].P_strt[t] * exp_pwr_price[t][hub] +
                                  z_ex[idx].Q_strt[t] * exp_ht_price[t][hub])

                for t in range(num_opt_var):
                    cost1 += 0.5 * delta * ((temp1[t])**2 + (temp2[t])**2)

                for idx in range(len(z_imp)):
                    if idx < hub:
                        for t in range(num_opt_var):
                            cost1 += (- z_imp[idx].P_strt[t] * imp_pwr_price[t][idx][hub-1] +
                                      0.5 * delta * (transfer_pw_imp[hub][idx][t] - z_imp[idx].P_strt[t])**2
                                      - z_imp[idx].Q_strt[t] * imp_ht_price[t][idx][hub-1] +
                                      0.5 * delta * (transfer_ht_imp[hub][idx][t] - z_imp[idx].Q_strt[t])**2)

                    if idx >= hub:
                        for t in range(num_opt_var):
                            cost1 += (- z_imp[idx].P_strt[t] * imp_pwr_price[t][idx+1][hub] +
                                      0.5 * delta * (transfer_pw_imp[hub][idx][t] - z_imp[idx].P_strt[t])**2
                                      - z_imp[idx].Q_strt[t] * imp_ht_price[t][idx+1][hub] +
                                      0.5 * delta * (transfer_ht_imp[hub][idx][t] - z_imp[idx].Q_strt[t])**2)


                # Solve with mosek or Gurobi
                problem = cp.Problem(cp.Minimize(cost1))
                problem.solve(solver=cp.MOSEK, verbose=True, save_file='opt_diagnosis.opf',
                              mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                                            mosek.dparam.optimizer_max_time: 200.0})

                opt_stat = problem.status
                opt_val = problem.value
                opt_time = problem.solver_stats
                print(f"Status:{opt_stat}, with Value:{opt_val:.2f}")

                temp1 = []
                temp2 = []

                for idx in range(len(z_imp)):
                    temp1.append(z_imp[idx].P_strt.value)
                    temp2.append(z_imp[idx].Q_strt.value)

                z_imp_pwr.append(temp1)
                z_imp_ht.append(temp2)

            z_pwr_final = np.array([[[0.00] * (num_hubs - 1)] * num_hubs] * num_opt_var)
            z_ht_final = np.array([[[0.00] * (num_hubs - 1)] * num_hubs] * num_opt_var)

            for hub in range(num_hubs):
                count = 0
                for idx in range(num_hubs):
                    if hub != idx:
                        if hub < idx:
                            for t in range(num_opt_var):
                                z_pwr_final[t][hub][count] = z_imp_pwr[idx][hub][t]
                                z_ht_final[t][hub][count] = z_imp_ht[idx][hub][t]
                        if idx < hub:
                            for t in range(num_opt_var):
                                z_pwr_final[t][hub][count] = z_imp_pwr[idx][hub-1][t]
                                z_ht_final[t][hub][count] = z_imp_ht[idx][hub-1][t]
                        count += 1

            transfer_pw_imp_final = np.array([[[0.00] * (num_hubs - 1)] * num_hubs] * num_opt_var)
            transfer_ht_imp_final = np.array([[[0.00] * (num_hubs - 1)] * num_hubs] * num_opt_var)

            for hub in range(num_hubs):
                count = 0
                for idx in range(num_hubs):
                    if hub != idx:
                        if hub < idx:
                            for t in range(num_opt_var):
                                transfer_pw_imp_final[t][hub][count] = transfer_pw_imp[idx][hub][t]
                                transfer_ht_imp_final[t][hub][count] = transfer_ht_imp[idx][hub][t]
                        if idx < hub:
                            for t in range(num_opt_var):
                                transfer_pw_imp_final[t][hub][count] = transfer_pw_imp[idx][hub-1][t]
                                transfer_ht_imp_final[t][hub][count] = transfer_ht_imp[idx][hub-1][t]
                        count += 1

            exp_pwr_price_new = np.array([[0.00] * num_hubs] * num_opt_var)
            exp_ht_price_new = np.array([[0.00] * num_hubs] * num_opt_var)
            imp_pwr_price_new = np.array([[[0.00] * (num_hubs - 1)] * num_hubs] * num_opt_var)
            imp_ht_price_new = np.array([[[0.00] * (num_hubs - 1)] * num_hubs] * num_opt_var)

            for hub in range(num_hubs):
                for t in range(num_opt_var):
                    exp_pwr_price_new[t][hub] = exp_pwr_price[t][hub]
                    exp_ht_price_new[t][hub] = exp_ht_price[t][hub]

            for hub in range(num_hubs):
                for idx in range(num_hubs - 1):
                    for t in range(num_opt_var):
                        exp_pwr_price_new[t][hub] -= delta * (transfer_pw_exp[hub][idx][t] -
                                                                  z_pwr_final[t][hub][idx])
                        exp_ht_price_new[t][hub] -= delta * (transfer_ht_exp[hub][idx][t] -
                                                                 z_ht_final[t][hub][idx])
            for hub in range(num_hubs):
                for idx in range(num_hubs-1):
                    for t in range(num_opt_var):
                        imp_pwr_price_new[t][hub][idx] = (imp_pwr_price[t][hub][idx] + delta *
                                                          (transfer_pw_imp_final[t][hub][idx] -
                                                           z_pwr_final[t][hub][idx]))
                        imp_ht_price_new[t][hub][idx] = (imp_ht_price[t][hub][idx] + delta *
                                                         (transfer_ht_imp_final[t][hub][idx] -
                                                          z_ht_final[t][hub][idx]))

            err = np.array([[0.0]*6]*num_opt_var)

            for t in range(num_opt_var):
                for hub in range(num_hubs):
                    err[t][0] += (exp_pwr_price_new[t][hub] - exp_pwr_price[t][hub])**2
                    err[t][1] += (exp_ht_price_new[t][hub] - exp_ht_price[t][hub])**2
                    for idx in range(num_hubs-1):
                        err[t][2] += (transfer_pw_imp_final[t][hub][idx] - transfer_pw_exp[hub][idx][t])**2
                        err[t][3] += (transfer_ht_imp_final[t][hub][idx] - transfer_ht_exp[hub][idx][t])**2
                        err[t][4] += (imp_pwr_price_new[t][hub][idx] - imp_pwr_price[t][hub][idx])**2
                        err[t][5] += (imp_ht_price_new[t][hub][idx] - imp_ht_price[t][hub][idx])**2