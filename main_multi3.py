# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mosek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
import time
import cvxpy as cp
import pickle
import os
import sys
import logging
from comp_classes_reduced import *
from test_multi import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - main - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('log_history.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

num_opt_var = 24
UPmin_CHP = 10
DNmin_CHP = 2

trcost = 0.01
P_delta = 0.1
Q_delta = 0.1
num_hubs: int = 3
BigM = 10 ** 10
hub = 1
beta = 0.003
d_PV = 0.15
d_PVT = 0.18
COP = 4.5
C_Fuel = 0.115
Tstc = 25
Tnoct = 48.3
Tstd = 20
n_CHP = 0.41
min_cap_CHP = 0.42
solar_area = 25700
tot = 0
delta = 0.04

exp_list, imp_list, hub_list = initTranlist(num_hubs)

time_start = dt.datetime(2018, 1, 1, 0, 0, 0)
time_end = dt.datetime(2018, 1, 1, 23, 0, 0)
time_now = time_start
start_time = dt.datetime(2018, 1, 1)

if __name__ == '__main__':



    capacities_conv, capacities_stor, tech_details, storage_details = import_capacities()
    network_cap = network_capacities()
    list_techs = []
    list_storage = []

    for hub in range(1, num_hubs + 1):
        list_techs.append(capacities_conv.loc[((capacities_conv.hub == hub) &
                                               (capacities_conv.value > 0)), 'tech'].values.tolist())
        list_storage.append(capacities_stor.loc[((capacities_stor.hub == hub) &
                                                 (capacities_stor.value > 0)), 'tech'].values.tolist())

    demand_data = get_data(start_time)


    def main_optimize(elec_storage, battery_depth, thermal_storage, CHP_Runtime, CHP_Downtime, capacities_conv,
                      capacities_stor, tech_details, storage_details, network_cap, list_techs, list_storage, lagran_P,
                      z_list_P, p_avg, lagran_H, z_list_H, h_avg, time_now):

        comp_start = time.perf_counter()

        cond = True
        iter = 0

        while cond:

            power = []
            heat = []
            grid_in = []
            grid_out = []
            storage_heatin = []
            storage_heatout = []
            storage_elecin = []
            storage_elecout = []
            tech_cost = []
            storage_cost = []
            grid_cost = []
            final_cost = []
            transfer_cost = []
            storage_elec = []
            storage_heat = []
            batt_depth = []
            CHP_ontime = []
            CHP_offtime = []
            demand_power = []
            demand_heat = []
            transfer_ht = np.array([[[0.0] * (num_hubs * (num_hubs - 1))] * num_hubs] * num_opt_var)
            transfer_pw = np.array([[[0.0] * (num_hubs * (num_hubs - 1))] * num_hubs] * num_opt_var)
            check_array = []

            for hub in range(num_hubs):

                cost = 0
                constr = []
                tran_list = []
                comp_list = []
                power_list = []
                heat_list = []
                it = 0

                hub_el = 'elec_' + str(hub + 1)
                hub_ht = 'heat_' + str(hub + 1)

                P_Demand = demand_data.loc[time_now: time_now + dt.timedelta(hours=23), hub_el].values.tolist()
                Q_Demand = demand_data.loc[time_now: time_now + dt.timedelta(hours=23), hub_ht].values.tolist()
                Ta = demand_data.loc[time_now: time_now + dt.timedelta(hours=23), 'temp'].values.tolist()
                I_Solar = demand_data.loc[time_now: time_now + dt.timedelta(hours=23), 'solar_roof'].values.tolist()

                Rini_CHP = CHP_Runtime[hub]
                Dini_CHP = CHP_Downtime[hub]

                count = 0
                ex_list = []
                im_list = []
                ch_list = []
                for rt in range(num_hubs):
                    for st in range(num_hubs):
                        if st != rt:
                            if rt == hub:
                                ex_list.append(count)
                                ch_list.append(1)
                                tran_list.append(trans_local(num_opt_var))
                                Pmax_tran = math.ceil(network_cap.loc[((network_cap.node1 == rt + 1) &
                                                                       (
                                                                               network_cap.node2 == st + 1)), 'value'].squeeze())
                                for t in range(num_opt_var):
                                    constr += [tran_list[len(tran_list) - 1].P_strt[t] >= 0,
                                               tran_list[len(tran_list) - 1].P_strt[t] <= Pmax_tran,
                                               tran_list[len(tran_list) - 1].Q_strt[t] >= 0,
                                               tran_list[len(tran_list) - 1].Q_strt[t] <= Pmax_tran]
                            elif st == hub:
                                im_list.append(count)
                                ch_list.append(1)
                                tran_list.append(trans_local(num_opt_var))
                                Pmax_tran = math.ceil(network_cap.loc[((network_cap.node1 == rt + 1) &
                                                                       (
                                                                               network_cap.node2 == st + 1)), 'value'].squeeze())
                                for t in range(num_opt_var):
                                    constr += [tran_list[len(tran_list) - 1].P_strt[t] >= 0,
                                               tran_list[len(tran_list) - 1].P_strt[t] <= Pmax_tran,
                                               tran_list[len(tran_list) - 1].Q_strt[t] >= 0,
                                               tran_list[len(tran_list) - 1].Q_strt[t] <= Pmax_tran]
                            else:
                                tran_list.append(0)
                                ch_list.append(0)

                            count += 1

                check_array.append(ch_list)

                Qmax_GSHP = 0
                hb = hub + 1

                for item in list_techs[hub]:

                    if item == 'solar_PV':
                        Pmax_PV = math.ceil(capacities_conv.loc[((capacities_conv.hub == hb) & (
                                capacities_conv.tech == item)), 'value'].squeeze())
                        Eff_PV = math.ceil(tech_details.loc[(tech_details.tech == item), 'eff'].squeeze()) * 0.01
                        Pmin_PV = 0

                        comp_list.append(PV(num_opt_var))

                        power_list.append(comp_list[it].P_PV)

                        for t in range(num_opt_var):
                            comp_list[it].I_PV[t] = I_Solar[t] * (Pmax_PV / d_PV)
                            comp_list[it].TempEff_PV[t] = 1 + ((-beta) * ((Ta[t] - Tstc) +
                                                                          (Tnoct - Ta[t]) * (I_Solar[t] / 0.8)))

                        for t in range(num_opt_var):
                            constr += [comp_list[it].P_PV[t] >= 0,
                                       comp_list[it].P_PV[t] <= (Eff_PV * comp_list[it].TempEff_PV[t] *
                                                                 comp_list[it].I_PV[t])]

                        it = it + 1

                    elif item == 'solar_PVT':

                        Pmax_PVT = math.ceil(capacities_conv.loc[((capacities_conv.hub == hb) & (
                                capacities_conv.tech == item)), 'value'].squeeze())
                        Eff_PVT = tech_details.loc[(tech_details.tech == item), 'outshare'].squeeze()
                        list1 = Eff_PVT.split(",")
                        li1 = []
                        for t in list1:
                            li1.append(float(t))

                        PEff_PVT = li1[0]
                        QEff_PVT = li1[1]

                        Eff_PVT = tech_details.loc[(tech_details.tech == item), 'eff'].squeeze() * 0.01

                        comp_list.append(PVT(num_opt_var))

                        power_list.append(comp_list[it].P_PVT)
                        heat_list.append(comp_list[it].Q_PVT)

                        for t in range(num_opt_var):
                            comp_list[it].I_PVT[t] = I_Solar[t] * (Pmax_PVT / d_PVT)
                            comp_list[it].TempEff_PVT[t] = 1 + ((-beta) * ((Ta[t] - Tstc) +
                                                                           (Tnoct - Ta[t]) * (I_Solar[t] / 0.8)))

                        for t in range(num_opt_var):
                            constr += [comp_list[it].P_PVT[t] >= 0, comp_list[it].Q_PVT[t] >= 0,
                                       comp_list[it].Out_PVT[t] <= (Eff_PVT * comp_list[it].TempEff_PVT[t] *
                                                                    comp_list[it].I_PVT[t]),
                                       comp_list[it].P_PVT[t] == PEff_PVT * comp_list[it].Out_PVT[t],
                                       comp_list[it].Q_PVT[t] == QEff_PVT * comp_list[it].Out_PVT[t]]

                        it = it + 1

                    elif item == 'Gas_CHP_unit_1':

                        Pmax_mCHP = math.ceil(capacities_conv.loc[((capacities_conv.hub == hb) & (
                                capacities_conv.tech == item)), 'value'].squeeze())
                        Pmin_mCHP = 0
                        Eff_mCHP = tech_details.loc[(tech_details.tech == item), 'outshare'].squeeze()

                        list2 = Eff_mCHP.split(",")
                        li = []
                        for k in list2:
                            li.append(float(k))
                        PEff_mCHP = li[0]
                        QEff_mCHP = li[1]
                        Eff_mCHP = tech_details.loc[(tech_details.tech == item), 'eff'].squeeze() * 0.01

                        comp_list.append(mCHP(num_opt_var))

                        power_list.append(comp_list[it].P_mCHP)
                        heat_list.append(comp_list[it].Q_mCHP)

                        for t in range(num_opt_var):
                            cost += comp_list[it].C_mCHP[t]
                            constr += [comp_list[it].Out_mCHP[t] >= Pmin_mCHP,
                                       comp_list[it].Out_mCHP[t] <= Pmax_mCHP,
                                       comp_list[it].P_mCHP[t] == PEff_mCHP * comp_list[it].Out_mCHP[t],
                                       comp_list[it].Q_mCHP[t] == QEff_mCHP * comp_list[it].Out_mCHP[t],
                                       comp_list[it].C_mCHP[t] == C_Fuel * comp_list[it].F_mCHP[t],
                                       comp_list[it].F_mCHP[t] == comp_list[it].P_mCHP[t] / Eff_mCHP]

                        it = it + 1

                    elif item == 'Gas_CHP_unit_2':

                        Pmax_CHP = math.ceil(capacities_conv.loc[((capacities_conv.hub == hb) & (
                                capacities_conv.tech == item)), 'value'].squeeze())
                        Pmin_CHP = 0
                        Eff_CHP = tech_details.loc[(tech_details.tech == item), 'outshare'].squeeze()
                        list2 = Eff_CHP.split(",")
                        li = []
                        for m in list2:
                            li.append(float(m))
                        PEff_CHP = li[0]
                        QEff_CHP = li[1]
                        Eff_CHP = tech_details.loc[(tech_details.tech == item), 'eff'].squeeze() * 0.01
                        CHP_points = CHP_operation(Pmax_CHP, Eff_CHP, PEff_CHP, QEff_CHP, n_CHP, min_cap_CHP)
                        CHP_fuelcap = Pmax_CHP / Eff_CHP

                        P11_CHP = CHP_points[1][0]
                        P12_CHP = CHP_points[1][0]
                        P13_CHP = CHP_points[1][3]
                        P14_CHP = CHP_points[1][4]
                        P21_CHP = CHP_points[1][0]
                        P22_CHP = CHP_points[1][1]
                        P23_CHP = CHP_points[1][2]
                        P24_CHP = CHP_points[1][3]

                        Q11_CHP = CHP_points[0][0]
                        Q12_CHP = CHP_points[0][0]
                        Q13_CHP = CHP_points[0][3]
                        Q14_CHP = CHP_points[0][4]
                        Q21_CHP = CHP_points[0][0]
                        Q22_CHP = CHP_points[0][1]
                        Q23_CHP = CHP_points[0][2]
                        Q24_CHP = CHP_points[0][3]

                        comp_list.append(CHP(num_opt_var))

                        power_list.append(comp_list[it].P_CHP)
                        heat_list.append(comp_list[it].Q_CHP)

                        constr += [comp_list[it].ysum_CHP[0] == comp_list[it].yon_CHP[0],
                                   comp_list[it].zsum_CHP[0] == comp_list[it].zoff_CHP[0],
                                   comp_list[it].R_CHP[0] == (Rini_CHP + 1) * comp_list[it].b_CHP[0],
                                   comp_list[it].D_CHP[0] == (Dini_CHP + 1) * (1 - comp_list[it].b_CHP[0])]

                        if Rini_CHP == 0:
                            constr += [comp_list[it].zoff_CHP[0] == 0,
                                       comp_list[it].yon_CHP[0] == comp_list[it].b_CHP[0]]
                        elif Dini_CHP == 0:
                            constr += [comp_list[it].zoff_CHP[0] == 1 - comp_list[it].b_CHP[0],
                                       comp_list[it].yon_CHP[0] == 0]

                        for t in range(num_opt_var):

                            cost += comp_list[it].C_CHP[t]
                            constr += [comp_list[it].P_CHP[t] <= Pmax_CHP * 10 * comp_list[it].b_CHP[t],
                                       comp_list[it].Q_CHP[t] <= Pmax_CHP * 10 * comp_list[it].b_CHP[t],
                                       comp_list[it].P_CHP[t] >= Pmin_CHP * comp_list[it].b_CHP[t],
                                       comp_list[it].Q_CHP[t] >= Pmin_CHP * comp_list[it].b_CHP[t],
                                       comp_list[it].P_CHP[t] == (
                                               comp_list[it].w11_CHP[t] * P11_CHP + comp_list[it].w12_CHP[t] * P12_CHP +
                                               comp_list[it].w13_CHP[t] * P13_CHP + comp_list[it].w14_CHP[t] * P14_CHP +
                                               comp_list[it].w21_CHP[t] * P21_CHP + comp_list[it].w22_CHP[t] * P22_CHP +
                                               comp_list[it].w23_CHP[t] * P23_CHP + comp_list[it].w24_CHP[t] * P24_CHP),
                                       comp_list[it].Q_CHP[t] == (
                                               comp_list[it].w11_CHP[t] * Q11_CHP + comp_list[it].w12_CHP[t] * Q12_CHP +
                                               comp_list[it].w13_CHP[t] * Q13_CHP + comp_list[it].w14_CHP[t] * Q14_CHP +
                                               comp_list[it].w21_CHP[t] * Q21_CHP + comp_list[it].w22_CHP[t] * Q22_CHP +
                                               comp_list[it].w23_CHP[t] * Q23_CHP + comp_list[it].w24_CHP[t] * Q24_CHP),
                                       comp_list[it].b1_CHP[t] + comp_list[it].b2_CHP[t] == comp_list[it].b_CHP[t],
                                       comp_list[it].w11_CHP[t] + comp_list[it].w12_CHP[t] + comp_list[it].w13_CHP[t] +
                                       comp_list[it].w14_CHP[t] == comp_list[it].b1_CHP[t],
                                       comp_list[it].w21_CHP[t] + comp_list[it].w22_CHP[t] + comp_list[it].w23_CHP[t] +
                                       comp_list[it].w24_CHP[t] == comp_list[it].b2_CHP[t],
                                       comp_list[it].w11_CHP[t] >= 0, comp_list[it].w12_CHP[t] >= 0,
                                       comp_list[it].w13_CHP[t] >= 0, comp_list[it].w14_CHP[t] >= 0,
                                       comp_list[it].w21_CHP[t] >= 0, comp_list[it].w22_CHP[t] >= 0,
                                       comp_list[it].w23_CHP[t] >= 0, comp_list[it].w24_CHP[t] >= 0,
                                       comp_list[it].w11_CHP[t] <= 1, comp_list[it].w12_CHP[t] <= 1,
                                       comp_list[it].w13_CHP[t] <= 1, comp_list[it].w14_CHP[t] <= 1,
                                       comp_list[it].w21_CHP[t] <= 1, comp_list[it].w22_CHP[t] <= 1,
                                       comp_list[it].w23_CHP[t] <= 1, comp_list[it].w24_CHP[t] <= 1,
                                       comp_list[it].yon_CHP[t] + comp_list[it].zoff_CHP[t] <= 1,
                                       comp_list[it].C_CHP[t] == C_Fuel * comp_list[it].F_CHP[t],
                                       comp_list[it].F_CHP[t] == comp_list[it].P_CHP[t] / Eff_CHP]
                            if t >= 1:
                                constr += [(comp_list[it].P_CHP[t] <= comp_list[it].P_CHP[t - 1] + 0.5 * Pmax_CHP *
                                            (comp_list[it].b_CHP[t - 1] + comp_list[it].yon_CHP[t])),
                                           (comp_list[it].P_CHP[t] >= comp_list[it].P_CHP[t - 1] - 0.5 * Pmax_CHP *
                                            (comp_list[it].b_CHP[t] + comp_list[it].zoff_CHP[t])),
                                           (comp_list[it].Q_CHP[t] <= comp_list[it].Q_CHP[t - 1] + 0.5 * Pmax_CHP *
                                            (comp_list[it].b_CHP[t - 1] + comp_list[it].yon_CHP[t])),
                                           (comp_list[it].Q_CHP[t] >= comp_list[it].Q_CHP[t - 1] - 0.5 * Pmax_CHP *
                                            (comp_list[it].b_CHP[t] + comp_list[it].zoff_CHP[t])),
                                           (comp_list[it].yon_CHP[t] - comp_list[it].zoff_CHP[t] ==
                                            comp_list[it].b_CHP[t] - comp_list[it].b_CHP[t - 1])]

                            # min up time constraints
                            if UPmin_CHP > 0:
                                if t >= UPmin_CHP:
                                    constr += [comp_list[it].b_CHP[t] >= comp_list[it].ysum_CHP[t],
                                               comp_list[it].ysum_CHP[t] == (comp_list[it].ysum_CHP[t - 1] -
                                                                             comp_list[it].yon_CHP[t - UPmin_CHP] +
                                                                             comp_list[it].yon_CHP[t])]

                                elif 1 <= t < UPmin_CHP:
                                    constr += [comp_list[it].b_CHP[t] >= comp_list[it].ysum_CHP[t],
                                               comp_list[it].ysum_CHP[t] == (comp_list[it].ysum_CHP[t - 1] +
                                                                             comp_list[it].yon_CHP[t])]

                                if 0 < Rini_CHP < UPmin_CHP:
                                    if t < (UPmin_CHP - Rini_CHP):
                                        constr += [comp_list[it].b_CHP[t] == 1,
                                                   comp_list[it].yon_CHP[t] == 0, comp_list[it].zoff_CHP[t] == 0]

                                elif Rini_CHP >= UPmin_CHP:
                                    if t == 0:
                                        constr += [comp_list[it].zoff_CHP[t] == 1 - comp_list[it].b_CHP[t],
                                                   comp_list[it].yon_CHP[t] == 0]

                            if DNmin_CHP > 0:

                                if t >= DNmin_CHP:
                                    constr += [(1 - comp_list[it].b_CHP[t]) >= comp_list[it].zsum_CHP[t],
                                               comp_list[it].zsum_CHP[t] == (comp_list[it].zsum_CHP[t - 1] -
                                                                             comp_list[it].zoff_CHP[t - DNmin_CHP] +
                                                                             comp_list[it].zoff_CHP[t])]

                                if 1 <= t < DNmin_CHP:
                                    constr += [(1 - comp_list[it].b_CHP[t]) >= comp_list[it].zsum_CHP[t],
                                               comp_list[it].zsum_CHP[t] == (comp_list[it].zsum_CHP[t - 1] +
                                                                             comp_list[it].zoff_CHP[t])]

                                if 0 < Dini_CHP < DNmin_CHP:
                                    if t < (DNmin_CHP - Dini_CHP):
                                        constr += [comp_list[it].b_CHP[t] == 0, comp_list[it].yon_CHP[t] == 0,
                                                   comp_list[it].zoff_CHP[t] == 0]

                                elif Dini_CHP >= DNmin_CHP:
                                    if t == 0:
                                        constr += [comp_list[it].zoff_CHP[t] == 0,
                                                   comp_list[it].yon_CHP[t] == comp_list[it].b_CHP[t]]

                        it = it + 1

                    elif item == 'GSHP_1' or item == 'GSHP_2':

                        Qmax_GSHP = math.ceil(capacities_conv.loc[((capacities_conv.hub == hb) &
                                                                   (capacities_conv.tech == item)), 'value'].squeeze())
                        Qmin_GSHP = 0

                        comp_list.append(GSHP(num_opt_var))

                        power_list.append(-comp_list[it].P_GSHP)
                        heat_list.append(comp_list[it].Q_GSHP)

                        for t in range(num_opt_var):
                            constr += [comp_list[it].Q_GSHP[t] <= Qmax_GSHP,
                                       comp_list[it].Q_GSHP[t] >= Qmin_GSHP,
                                       comp_list[it].Q_GSHP[t] == comp_list[it].P_GSHP[t] * COP]

                        it = it + 1

                    elif item == 'gas_boiler_1' or item == 'gas_boiler_2':

                        Qmax_GB = math.ceil(capacities_conv.loc[((capacities_conv.hub == hb) &
                                                                 (capacities_conv.tech == item)), 'value'].squeeze())
                        Qmin_GB = 0

                        x0_GB = 0
                        x1_GB = 0.25 * Qmax_GB
                        x2_GB = 0.5 * Qmax_GB
                        x3_GB = 0.75 * Qmax_GB
                        x4_GB = Qmax_GB

                        Eff1_GB = 0.01 * (21.49 + (182.18 * (1 / 4)) + ((-120.67) * (1 / 4) ** 2))
                        Eff2_GB = 0.01 * (21.49 + (182.18 * (2 / 4)) + ((-120.67) * (2 / 4) ** 2))
                        Eff3_GB = 0.01 * (21.49 + (182.18 * (3 / 4)) + ((-120.67) * (3 / 4) ** 2))
                        Eff4_GB = 0.01 * (21.49 + (182.18 * (4 / 4)) + ((-120.67) * (4 / 4) ** 2))

                        comp_list.append(GB(num_opt_var))

                        heat_list.append(comp_list[it].Q_GB)

                        for t in range(num_opt_var):
                            cost += comp_list[it].C_GB[t]
                            constr += [comp_list[it].Q_GB[t] <= comp_list[it].b_GB[t] * Qmax_GB,
                                       comp_list[it].Q_GB[t] >= comp_list[it].b_GB[t] * Qmin_GB,
                                       comp_list[it].Q_GB[t] == (comp_list[it].w1_GB[t] * x1_GB +
                                                                 comp_list[it].w2_GB[t] * x2_GB +
                                                                 comp_list[it].w3_GB[t] * x3_GB +
                                                                 comp_list[it].w4_GB[t] * x4_GB),
                                       (comp_list[it].b1_GB[t] + comp_list[it].b2_GB[t] +
                                        comp_list[it].b3_GB[t] + comp_list[it].b4_GB[t]) == comp_list[it].b_GB[t],
                                       (comp_list[it].w0_GB[t] + comp_list[it].w1_GB[t] + comp_list[it].w2_GB[t] +
                                        comp_list[it].w3_GB[t] + comp_list[it].w4_GB[t]) == comp_list[it].b_GB[t],
                                       comp_list[it].w0_GB[t] <= comp_list[it].b1_GB[t],
                                       comp_list[it].w1_GB[t] <= comp_list[it].b1_GB[t] + comp_list[it].b2_GB[t],
                                       comp_list[it].w2_GB[t] <= comp_list[it].b3_GB[t] + comp_list[it].b2_GB[t],
                                       comp_list[it].w3_GB[t] <= comp_list[it].b3_GB[t] + comp_list[it].b4_GB[t],
                                       comp_list[it].w4_GB[t] <= comp_list[it].b4_GB[t],
                                       comp_list[it].w0_GB[t] >= 0, comp_list[it].w1_GB[t] >= 0,
                                       comp_list[it].w2_GB[t] >= 0,
                                       comp_list[it].w3_GB[t] >= 0, comp_list[it].w4_GB[t] >= 0,
                                       comp_list[it].w0_GB[t] <= 1, comp_list[it].w1_GB[t] <= 1,
                                       comp_list[it].w2_GB[t] <= 1,
                                       comp_list[it].w3_GB[t] <= 1, comp_list[it].w4_GB[t] <= 1,
                                       comp_list[it].C_GB[t] == C_Fuel * comp_list[it].F_GB[t],
                                       comp_list[it].F_GB[t] == (comp_list[it].w1_GB[t] * (x1_GB / Eff1_GB) +
                                                                 comp_list[it].w2_GB[t] * (x2_GB / Eff2_GB) +
                                                                 comp_list[it].w3_GB[t] * (x3_GB / Eff3_GB) +
                                                                 comp_list[it].w4_GB[t] * (x4_GB / Eff4_GB))]

                        it = it + 1

                for item in list_storage[hub]:

                    if item == 'heat_storage':

                        Qmax_Storage = capacities_stor.loc[((capacities_stor.hub == hb) &
                                                            (capacities_stor.tech == item)), 'value'].squeeze()
                        Eff_Storage = storage_details.loc[(storage_details.tech == item), 'stateff'].squeeze()
                        Eff_StorageCh = storage_details.loc[(storage_details.tech == item), 'cyceff'].squeeze()
                        Eff_StorageDc = Eff_StorageCh

                        comp_list.append(Heat_Storage(num_opt_var))

                        constr += [comp_list[it].Q_StorageTot[0] == (Eff_Storage * thermal_storage[hub] +
                                                                     Eff_StorageCh * comp_list[it].Q_StorageCh[0] -
                                                                     (1 / Eff_StorageDc) * comp_list[it].Q_StorageDc[
                                                                         0])]

                        heat_list.append(comp_list[it].Q_StorageDc)
                        heat_list.append(-comp_list[it].Q_StorageCh)

                        for t in range(num_opt_var):

                            constr += [comp_list[it].Q_StorageCh[t] >= 0,
                                       comp_list[it].Q_StorageCh[t] <= BigM * comp_list[it].b_StorageCh[t],
                                       comp_list[it].Q_StorageDc[t] >= 0,
                                       comp_list[it].Q_StorageDc[t] <= BigM * (1 - comp_list[it].b_StorageCh[t]),
                                       comp_list[it].Q_StorageTot[t] >= 0.2 * Qmax_Storage,
                                       comp_list[it].Q_StorageTot[t] <= Qmax_Storage,
                                       comp_list[it].b_StorageCh[t] >= 0, comp_list[it].b_StorageCh[t] <= 1]

                            if t >= 1:
                                constr += [(comp_list[it].Q_StorageTot[t] ==
                                            Eff_Storage * comp_list[it].Q_StorageTot[t - 1] +
                                            Eff_StorageCh * comp_list[it].Q_StorageCh[t] -
                                            (1 / Eff_StorageDc) * comp_list[it].Q_StorageDc[t])]

                        it = it + 1

                    elif item == 'Battery':

                        Pmax_Battery = capacities_stor.loc[((capacities_stor.hub == hb) &
                                                            (capacities_stor.tech == item)), 'value'].squeeze()
                        Pmin_Battery = 0
                        Eff_Battery = storage_details.loc[(storage_details.tech == item), 'stateff'].squeeze()
                        Eff_BatteryCh = storage_details.loc[(storage_details.tech == item), 'cyceff'].squeeze()
                        Eff_BatteryDc = Eff_BatteryCh

                        comp_list.append(Elec_Storage(num_opt_var))

                        power_list.append(comp_list[it].P_BatteryDc)
                        power_list.append(-comp_list[it].P_BatteryCh)

                        constr += [comp_list[it].P_BatteryTot[0] == (Eff_Battery * elec_storage[hub] +
                                                                     Eff_BatteryCh * comp_list[it].P_BatteryCh[0] -
                                                                     (1 / Eff_BatteryDc) * comp_list[it].P_BatteryDc[
                                                                         0])]

                        for t in range(num_opt_var):

                            # cost += comp_list[it].C_Battery[t]
                            constr += [comp_list[it].P_BatteryTot[t] >= 0.2 * Pmax_Battery,
                                       comp_list[it].P_BatteryTot[t] <= 0.8 * Pmax_Battery,
                                       comp_list[it].P_BatteryCh[t] >= 0,
                                       comp_list[it].P_BatteryCh[t] <= BigM * comp_list[it].b_BatteryCh[t],
                                       comp_list[it].P_BatteryDc[t] >= 0,
                                       comp_list[it].P_BatteryDc[t] <= BigM * (1 - comp_list[it].b_BatteryCh[t]),
                                       comp_list[it].b_BatteryCh[t] >= 0, comp_list[it].b_BatteryCh[t] <= 1]

                            if t >= 1:
                                constr += [comp_list[it].P_BatteryTot[t] ==
                                           Eff_Battery * comp_list[it].P_BatteryTot[t - 1] +
                                           Eff_BatteryCh * comp_list[it].P_BatteryCh[t] -
                                           (1 / Eff_BatteryDc) * comp_list[it].P_BatteryDc[t]]

                        it = it + 1

                R_GridOut = demand_data.loc[time_now: time_now + dt.timedelta(hours=23), 'el_tariff'].values.tolist()
                R_GridIn = demand_data.loc[time_now: time_now + dt.timedelta(hours=23),
                           'feed_in_tariff'].values.tolist()

                comp_list.append(Elec_Grid(num_opt_var))
                power_list.append(comp_list[it].P_GridOut)
                power_list.append(-comp_list[it].P_GridIn)

                for idx in range(len(ex_list)):
                    power_list.append(-tran_list[int(ex_list[idx])].P_strt)
                    heat_list.append(-tran_list[int(ex_list[idx])].Q_strt)
                    for t in range(num_opt_var):
                        cost += (trcost * tran_list[int(ex_list[idx])].P_strt[t] +
                                 lagran_P[t][hub][int(ex_list[idx])] * (tran_list[int(ex_list[idx])].P_strt[t]) +
                                 delta * 0.5 * (tran_list[int(ex_list[idx])].P_strt[t] -
                                                z_list_P[t][int(ex_list[idx])]) ** 2)

                        cost += (trcost * tran_list[int(ex_list[idx])].Q_strt[t] +
                                 lagran_H[t][hub][int(ex_list[idx])] * (tran_list[int(ex_list[idx])].Q_strt[t]) +
                                 delta * 0.5 * (tran_list[int(ex_list[idx])].Q_strt[t] -
                                                z_list_H[t][int(ex_list[idx])]) ** 2)

                for idx in range(len(im_list)):
                    power_list.append(tran_list[int(im_list[idx])].P_strt * 0.95)
                    heat_list.append(tran_list[int(im_list[idx])].Q_strt * 0.90)
                    for t in range(num_opt_var):
                        cost += (lagran_P[t][hub][int(im_list[idx])] * (tran_list[int(im_list[idx])].P_strt[t]) +
                                 delta * 0.5 * (tran_list[int(im_list[idx])].P_strt[t] -
                                                z_list_P[t][int(im_list[idx])]) ** 2)

                        cost += (lagran_H[t][hub][int(im_list[idx])] * (tran_list[int(im_list[idx])].Q_strt[t]) +
                                 delta * 0.5 * (tran_list[int(im_list[idx])].Q_strt[t] -
                                                z_list_H[t][int(im_list[idx])]) ** 2)

                if len(power_list) < 35:
                    lnp = len(power_list)
                    while lnp <= 34:
                        power_list.append([0] * num_opt_var)
                        lnp += 1

                if len(heat_list) < 35:
                    lnp = len(heat_list)
                    while lnp <= 34:
                        heat_list.append([0] * num_opt_var)
                        lnp += 1

                for t in range(num_opt_var):
                    # Demand
                    cost += comp_list[it].C_Grid[t]

                    constr += [P_Demand[t] == (power_list[0][t] + power_list[1][t] + power_list[2][t] +
                                               power_list[3][t] + power_list[4][t] + power_list[5][t] +
                                               power_list[6][t] + power_list[7][t] + power_list[8][t] +
                                               power_list[9][t] + power_list[10][t] + power_list[11][t] +
                                               power_list[12][t] + power_list[13][t] + power_list[14][t] +
                                               power_list[15][t] + power_list[16][t] + power_list[17][t] +
                                               power_list[18][t] + power_list[19][t] + power_list[20][t] +
                                               power_list[21][t] + power_list[22][t] + power_list[23][t] +
                                               power_list[24][t] + power_list[25][t] + power_list[26][t] +
                                               power_list[27][t] + power_list[28][t] + power_list[29][t] +
                                               power_list[30][t] + power_list[31][t] + power_list[32][t] +
                                               power_list[33][t] + power_list[34][t]),
                               Q_Demand[t] == (heat_list[0][t] + heat_list[1][t] + heat_list[2][t] +
                                               heat_list[3][t] + heat_list[4][t] + heat_list[5][t] +
                                               heat_list[6][t] + heat_list[7][t] + heat_list[8][t] +
                                               heat_list[9][t] + heat_list[10][t] + heat_list[11][t] +
                                               heat_list[12][t] + heat_list[13][t] + heat_list[14][t] +
                                               heat_list[15][t] + heat_list[16][t] + heat_list[17][t] +
                                               heat_list[18][t] + heat_list[19][t] + heat_list[20][t] +
                                               heat_list[21][t] + heat_list[22][t] + heat_list[23][t] +
                                               heat_list[24][t] + heat_list[25][t] + heat_list[26][t] +
                                               heat_list[27][t] + heat_list[28][t] + heat_list[29][t] +
                                               heat_list[30][t] + heat_list[31][t] + heat_list[32][t] +
                                               heat_list[33][t] + heat_list[34][t])]

                    constr += [comp_list[it].P_GridIn[t] >= 0,
                               comp_list[it].P_GridIn[t] <= BigM,
                               comp_list[it].P_GridOut[t] >= 0,
                               comp_list[it].P_GridOut[t] <= BigM,
                               comp_list[it].C_Grid[t] == (R_GridOut[t] * comp_list[it].P_GridOut[t] -
                                                           R_GridIn[t] * comp_list[it].P_GridIn[t]),
                               comp_list[it].P_Slack[t] >= 0, comp_list[it].Q_Slack[t] >= 0]

                it = it + 1

                demand_power.append(P_Demand[0])
                demand_heat.append(Q_Demand[0])

                problem = cp.Problem(cp.Minimize(cost), constr)

                # problem.solve(solver=cp.MOSEK, verbose=True, save_file='opt_diagnosis.opf',
                #               mosek_params={mosek.dparam.optimizer_max_time: 500.0})

                problem.solve(solver=cp.GUROBI, verbose=True)
                opt_stat = problem.status
                opt_val = problem.value

                print(f"Status:{opt_stat}, with Value:{opt_val:.2f}")

                # if opt_stat == 'infeasible':
                #     problem.solve(solver=cp.MOSEK, verbose=True, save_file='opt_diagnosis.opf',
                #                   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                #                                 mosek.dparam.optimizer_max_time: 100.0})
                it = 0

                pwr = []
                ht = []
                g_in = []
                g_out = []
                st_heatin = []
                st_heatout = []
                st_elecin = []
                st_elecout = []
                g_cost = []
                s_cost = 0
                t_cost = 0
                f_cost = 0
                tr_cost = 0
                st_heat = []
                st_elec = []
                C_on = []
                C_off = []
                tr_pw = []
                tr_ht = []

                for item in list_techs[hub]:

                    if item == 'solar_PV':
                        pwr.append(comp_list[it].P_PV.value[0])
                        ht.append(0)
                        it = it + 1

                    if item == 'solar_PVT':
                        pwr.append(comp_list[it].P_PVT.value[0])
                        ht.append(comp_list[it].Q_PVT.value[0])
                        it = it + 1

                    if item == 'Gas_CHP_unit_1':
                        pwr.append(comp_list[it].P_mCHP.value[0])
                        ht.append(comp_list[it].Q_mCHP.value[0])
                        t_cost += comp_list[it].C_mCHP.value[0]
                        f_cost += comp_list[it].C_mCHP.value[0]
                        it = it + 1

                    if item == 'Gas_CHP_unit_2':
                        pwr.append(comp_list[it].P_CHP.value[0])
                        ht.append(comp_list[it].Q_CHP.value[0])
                        t_cost += comp_list[it].C_CHP.value[0]

                        C_on.append(round(comp_list[it].R_CHP.value[0]))
                        C_off.append(round(comp_list[it].D_CHP.value[0]))
                        f_cost += comp_list[it].C_CHP.value[0]
                        it = it + 1

                    if item == 'GSHP_1' or item == 'GSHP_2':
                        pwr.append(-comp_list[it].P_GSHP.value[0])
                        ht.append(comp_list[it].Q_GSHP.value[0])
                        it = it + 1

                    if item == 'gas_boiler_1' or item == 'gas_boiler_2':
                        pwr.append(0)
                        ht.append(comp_list[it].Q_GB.value[0])
                        t_cost += comp_list[it].C_GB.value[0]
                        f_cost += comp_list[it].C_GB.value[0]
                        it = it + 1

                for item in list_storage[hub]:

                    if item == 'heat_storage':
                        st_heatin.append(-comp_list[it].Q_StorageCh.value[0])
                        st_heatout.append(comp_list[it].Q_StorageDc.value[0])
                        st_heat.append(comp_list[it].Q_StorageTot.value[0])
                        it = it + 1

                    if item == 'Battery':
                        st_elecin.append(-comp_list[it].P_BatteryCh.value[0])
                        st_elecout.append(comp_list[it].P_BatteryDc.value[0])
                        st_elec.append(comp_list[it].P_BatteryTot.value[0])

                        it = it + 1

                g_cost = comp_list[it].C_Grid.value[0]
                f_cost += comp_list[it].C_Grid.value[0]

                for idx in range(len(ex_list)):
                    tr_cost += trcost * (tran_list[int(ex_list[idx])].P_strt.value[0] +
                                         tran_list[int(ex_list[idx])].Q_strt.value[0])

                f_cost += tr_cost

                if not st_heatin:
                    storage_heatin.append(0.0)
                else:
                    storage_heatin.extend(st_heatin)

                if not st_heatout:
                    storage_heatout.append(0.0)
                else:
                    storage_heatout.extend(st_heatout)

                if not st_elecin:
                    storage_elecin.append(0.0)
                else:
                    storage_elecin.extend(st_elecin)

                if not st_elecout:
                    storage_elecout.append(0.0)
                else:
                    storage_elecout.extend(st_elecout)

                if not st_heat:
                    storage_heat.append(0.0)
                else:
                    storage_heat.extend(st_heat)

                if not st_elec:
                    storage_elec.append(0.0)
                else:
                    storage_elec.extend(st_elec)

                if not C_on:
                    CHP_ontime.append(0)
                else:
                    CHP_ontime.extend(C_on)

                if not C_off:
                    CHP_offtime.append(0)
                else:
                    CHP_offtime.extend(C_off)

                grid_in.append(-comp_list[it].P_GridIn.value[0])
                grid_out.append(comp_list[it].P_GridOut.value[0])

                power.append(pwr)
                heat.append(ht)

                grid_cost.append(g_cost)
                final_cost.append(f_cost)
                tech_cost.append(t_cost)
                storage_cost.append(s_cost)
                transfer_cost.append(tr_cost)

                for idx in range(len(tran_list)):
                    if idx in ex_list or idx in im_list:
                        for t in range(num_opt_var):
                            transfer_pw[t][hub][idx] = tran_list[idx].P_strt.value[t]
                            transfer_ht[t][hub][idx] = tran_list[idx].Q_strt.value[t]

            p_avg_new = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            z_P_new = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            lagran_P_new = np.array([[[0.0] * int(num_hubs * (num_hubs - 1))] * int(num_hubs)] * num_opt_var)

            h_avg_new = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            z_H_new = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            lagran_H_new = np.array([[[0.0] * int(num_hubs * (num_hubs - 1))] * int(num_hubs)] * num_opt_var)

            for hb in range(num_hubs):
                for idx in range(int(num_hubs * (num_hubs - 1))):
                    if check_array[hb][idx] == 1:
                        for t in range(num_opt_var):
                            p_avg_new[t][idx] += 0.5 * transfer_pw[t][hb][idx]
                            z_P_new[t][idx] += 0.5 * transfer_pw[t][hb][idx] + 0.5 * (1 / delta) * lagran_P[t][hb][idx]

                            h_avg_new[t][idx] += 0.5 * transfer_ht[t][hb][idx]
                            z_H_new[t][idx] += 0.5 * transfer_ht[t][hb][idx] + 0.5 * (1 / delta) * lagran_H[t][hb][idx]

            for hb in range(num_hubs):
                for idx in range(int(num_hubs * (num_hubs - 1))):
                    if check_array[hb][idx] == 1:
                        for t in range(num_opt_var):
                            lagran_P_new[t][hb][idx] = lagran_P[t][hb][idx] + delta * (transfer_pw[t][hb][idx] -
                                                                                       z_P_new[t][idx])

                            lagran_H_new[t][hb][idx] = lagran_H[t][hb][idx] + delta * (transfer_ht[t][hb][idx] -
                                                                                       z_H_new[t][idx])

            prim_P_resid = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            dual_P_resid = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)

            prim_H_resid = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            dual_H_resid = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)

            for hb in range(num_hubs):
                for idx in range(int(num_hubs * (num_hubs - 1))):
                    if check_array[hb][idx] == 1:
                        for t in range(num_opt_var):
                            prim_P_resid[t][idx] += (transfer_pw[t][hb][idx] - p_avg_new[t][idx]) ** 2
                            prim_H_resid[t][idx] += (transfer_ht[t][hb][idx] - h_avg_new[t][idx]) ** 2

            for idx in range(int(num_hubs * (num_hubs - 1))):
                for t in range(num_opt_var):
                    dual_P_resid[t][idx] = 2 * (delta ** 2) * (p_avg_new[t][idx] - p_avg[t][idx]) ** 2
                    dual_H_resid[t][idx] = 2 * (delta ** 2) * (h_avg_new[t][idx] - h_avg[t][idx]) ** 2

            dual_P_2norm = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            prim_P_2norm = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)

            dual_H_2norm = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
            prim_H_2norm = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)

            cond = False
            count = [0] * num_opt_var
            for idx in range(int(num_hubs * (num_hubs - 1))):
                for t in range(num_opt_var):
                    dual_P_2norm[t][idx] = math.sqrt(dual_P_resid[t][idx])
                    prim_P_2norm[t][idx] = math.sqrt(prim_P_resid[t][idx])
                    dual_H_2norm[t][idx] = math.sqrt(dual_H_resid[t][idx])
                    prim_H_2norm[t][idx] = math.sqrt(prim_H_resid[t][idx])
                    if prim_H_2norm[t][idx] >= 0.1 or prim_P_2norm[t][idx] >= 0.1:
                        cond = True
                        count[t] += 1

            for idx in range(int(num_hubs * (num_hubs - 1))):
                for t in range(num_opt_var):
                    h_avg[t][idx] = h_avg_new[t][idx]
                    p_avg[t][idx] = p_avg_new[t][idx]
                    z_list_P[t][idx] = z_P_new[t][idx]
                    z_list_H[t][idx] = z_H_new[t][idx]

            for hb in range(num_hubs):
                for idx in range(int(num_hubs * (num_hubs - 1))):
                    for t in range(num_opt_var):
                        lagran_P[t][hb][idx] = lagran_P_new[t][hb][idx]
                        lagran_H[t][hb][idx] = lagran_H_new[t][hb][idx]

            iter += 1

            if iter >= 60:
                cond = False

        comp_end = time.perf_counter()
        print(comp_end - comp_start)
        print('run complete')

        return demand_power, demand_heat, power, heat, grid_in, grid_out, storage_heatin, storage_heatout, \
               storage_elecin, storage_elecout, tech_cost, storage_cost, grid_cost, final_cost, transfer_cost, \
               storage_heat, storage_elec, CHP_ontime, CHP_offtime, p_avg, z_list_P, lagran_P, h_avg, z_list_H, \
               lagran_H, iter


    Battery_max = []
    elec_storage = []
    Thermal_max = []
    thermal_storage = []
    battery_depth = []

    for hub in range(1, num_hubs + 1):
        if 'Battery' in list_storage[hub - 1]:
            temp = capacities_stor.loc[((capacities_stor.hub == hub) &
                                        (capacities_stor.tech == 'Battery')), 'value'].squeeze()
            Battery_max.append(temp)
            elec_storage.append(temp * 0.25)

            b_depth = [0, 0, 0, 0]
            if temp > 0:
                for j in range(4):
                    bat_SOC = elec_storage[hub - 1] / temp
                    if bat_SOC >= ((j + 1) * 0.2):
                        b_depth[j] = 0.2

                    elif bat_SOC < ((j + 1) * 0.2) and (bat_SOC >= 0.2 * j):

                        b_depth[j] = bat_SOC - 0.2 * j

                    else:
                        b_depth[j] = 0.0

            battery_depth.append(b_depth)

        else:

            Battery_max.append(0)
            elec_storage.append(0)
            battery_depth.append([0, 0, 0, 0])

        if 'heat_storage' in list_storage[hub - 1]:
            temp = capacities_stor.loc[((capacities_stor.hub == hub) &
                                        (capacities_stor.tech == 'heat_storage')), 'value'].squeeze()
            Thermal_max.append(temp)
            thermal_storage.append(temp * 0.25)
        else:
            Thermal_max.append(0)
            thermal_storage.append(0)

    CHP_Runtime = [0, 0, 0]
    CHP_Downtime = [12, 0, 0]

    p_avg_in = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
    z_list_P_in = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
    lagran_P_in = np.array([[[0.0] * int(num_hubs * (num_hubs - 1))] * int(num_hubs)] * num_opt_var)

    h_avg_in = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
    z_list_H_in = np.array([[0.0] * int(num_hubs * (num_hubs - 1))] * num_opt_var)
    lagran_H_in = np.array([[[0.0] * int(num_hubs * (num_hubs - 1))] * int(num_hubs)] * num_opt_var)

    demand_power_final = []
    demand_heat_final = []
    power_final = []
    heat_final = []
    grid_in_final = []
    grid_out_final = []
    transfer_cost_final = []
    storage_heatin_final = []
    storage_heatout_final = []
    storage_elecin_final = []
    storage_elecout_final = []

    tech_cost_final = []
    storage_cost_final = []
    final_cost_final = []
    grid_cost_final = []
    storage_elec_final = []
    storage_heat_final = []
    CHP_ontime_final = []
    CHP_offtime_final = []
    SOC_elec = []
    SOC_therm = []
    date_info = []
    time_info = []
    p_avg_final = []
    z_list_P_final = []
    lagran_P_final = []

    h_avg_final = []
    z_list_H_final = []
    lagran_H_final = []
    iter_final = []

    time_now -= dt.timedelta(hours=1)

    while time_now < time_end:
        time_now += dt.timedelta(hours=1)
        logger.info(time_now)
        # Optimization based on current state, model and forecast of boundary conditions
        (demand_power, demand_heat, power, heat, grid_in, grid_out, storage_heatin, storage_heatout, storage_elecin,
         storage_elecout, tech_cost, storage_cost, grid_cost, final_cost, transfer_cost, storage_heat, storage_elec,
         CHP_ontime, CHP_offtime, p_avg, z_list_P, lagran_P, h_avg,
         z_list_H, lagran_H, itera) = main_optimize(elec_storage, battery_depth, thermal_storage, CHP_Runtime,
                                                    CHP_Downtime, capacities_conv, capacities_stor, tech_details,
                                                    storage_details, network_cap, list_techs, list_storage,
                                                    lagran_P_in, z_list_P_in, p_avg_in, lagran_H_in, z_list_H_in,
                                                    h_avg_in, time_now)

        demand_power_final.append(demand_power)
        demand_heat_final.append(demand_heat)
        power_final.append(power)
        heat_final.append(heat)
        grid_in_final.append(grid_in)
        grid_out_final.append(grid_out)
        storage_heatin_final.append(storage_heatin)
        storage_heatout_final.append(storage_heatout)
        storage_elecin_final.append(storage_elecin)
        storage_elecout_final.append(storage_elecout)

        p_avg_final.append(p_avg[0].tolist())
        z_list_P_final.append(z_list_P[0].tolist())
        lagran_P_final.append(lagran_P[0].tolist())

        h_avg_final.append(h_avg[0].tolist())
        z_list_H_final.append(z_list_H[0].tolist())
        lagran_H_final.append(lagran_H[0].tolist())

        iter_final.append(itera)

        tech_cost_final.append(tech_cost)
        storage_cost_final.append(storage_cost)
        final_cost_final.append(final_cost)
        grid_cost_final.append(grid_cost)
        transfer_cost_final.append(transfer_cost)

        storage_heat_final.append(storage_heat)
        storage_elec_final.append(storage_elec)

        CHP_ontime_final.append(CHP_ontime)
        CHP_offtime_final.append(CHP_offtime)

        SOC_e = []
        SOC_th = []

        for hub in range(0, num_hubs):
            if Battery_max[hub] > 0:
                temp = storage_elec[hub] * 100 / Battery_max[hub]
                SOC_e.append(temp)
            else:
                SOC_e.append(0)

            if Thermal_max[hub] > 0:
                temp1 = storage_heat[hub] * 100 / Thermal_max[hub]
                SOC_th.append(temp1)
            else:
                SOC_th.append(0)

        SOC_elec.append(SOC_e)
        SOC_therm.append(SOC_th)

        str1 = str(time_now.day) + "." + str(time_now.month) + "." + str(time_now.year)
        str2 = str(time_now.hour) + ":" + str(time_now.minute) + ":" + str(time_now.second)
        date_info.append([str1])
        time_info.append([str2])

        elec_storage = storage_elec
        battery_depth = battery_depth
        thermal_storage = storage_heat
        CHP_Runtime = CHP_ontime
        CHP_Downtime = CHP_offtime

        for t in range(num_opt_var - 1):
            p_avg_in[t] = p_avg[t + 1].tolist()
            z_list_P_in[t] = z_list_P[t + 1].tolist()
            lagran_P_in[t] = lagran_P[t + 1].tolist()

            h_avg_in[t] = h_avg[t + 1].tolist()
            z_list_H_in[t] = z_list_H[t + 1].tolist()
            lagran_H_in[t] = lagran_H[t + 1].tolist()

        # p_avg_in[num_opt_var - 1] = p_avg[num_opt_var - 1].tolist()
        # z_list_P_in[num_opt_var - 1] = z_list_P[num_opt_var - 1].tolist()
        # lagran_P_in[num_opt_var - 1] = lagran_P[num_opt_var - 1].tolist()
        #
        # h_avg_in[num_opt_var - 1] = h_avg[num_opt_var - 1].tolist()
        # z_list_H_in[num_opt_var - 1] = z_list_H[num_opt_var - 1].tolist()
        # lagran_H_in[num_opt_var - 1] = lagran_H[num_opt_var - 1].tolist()
        #
        p_avg_in[num_opt_var - 1] = p_avg[0].tolist()
        z_list_P_in[num_opt_var - 1] = z_list_P[0].tolist()
        lagran_P_in[num_opt_var - 1] = lagran_P[0].tolist()

        h_avg_in[num_opt_var - 1] = h_avg[0].tolist()
        z_list_H_in[num_opt_var - 1] = z_list_H[0].tolist()
        lagran_H_in[num_opt_var - 1] = lagran_H[0].tolist()

        # p_avg_in[num_opt_var - 1] = np.array([0.0] * int(num_hubs * (num_hubs - 1))).tolist()
        # z_list_P_in[num_opt_var - 1] = np.array([0.0] * int(num_hubs * (num_hubs - 1))).tolist()
        # lagran_P_in[num_opt_var - 1] = np.array([0.0] * int(num_hubs * (num_hubs - 1))).tolist()
        #
        # h_avg_in[num_opt_var - 1] = np.array([0.0] * int(num_hubs * (num_hubs - 1))).tolist()
        # z_list_H_in[num_opt_var - 1] = np.array([0.0] * int(num_hubs * (num_hubs - 1))).tolist()
        # lagran_H_in[num_opt_var - 1] = np.array([0.0] * int(num_hubs * (num_hubs - 1))).tolist()

        print("wait")

    with xlsxwriter.Workbook('results_12res_diffeps.xlsx') as workbook:

        savePower(workbook, demand_power_final, power_final, grid_in_final, grid_out_final, storage_elecin_final,
                  storage_elecout_final, p_avg_final, list_techs, list_storage, date_info, time_info, num_hubs)

        saveHeat(workbook, demand_heat_final, heat_final, storage_heatin_final, storage_heatout_final, h_avg_final,
                 list_techs, list_storage, date_info, time_info, num_hubs)

        saveCost(workbook, tech_cost_final, storage_cost_final, final_cost_final, grid_cost_final,
                 transfer_cost_final, date_info, time_info, num_hubs)

        saveStorage(workbook, storage_heat_final, storage_elec_final, SOC_elec, SOC_therm, storage_elecin_final,
                    storage_elecout_final, storage_heatin_final, storage_heatout_final, list_storage, date_info,
                    time_info, num_hubs)

        saveChp(workbook, CHP_ontime_final, CHP_offtime_final, list_techs, date_info, time_info, num_hubs)

        savetrans(workbook, p_avg_final, z_list_P_final, lagran_P_final, h_avg_final, z_list_H_final, lagran_H_final,
                  date_info, time_info, num_hubs)

    print('complete and saved')

