import cvxpy as cp


class PV:
    def __init__(self, num_opt_var):
        self.P_PV = cp.Variable(num_opt_var)
        self.I_PV = [0] * num_opt_var
        self.TempEff_PV = [0] * num_opt_var


class PVT:
    def __init__(self, num_opt_var):
        self.P_PVT = cp.Variable(num_opt_var)
        self.Q_PVT = cp.Variable(num_opt_var)
        self.Out_PVT = cp.Variable(num_opt_var)
        self.I_PVT = [0] * num_opt_var
        self.TempEff_PVT = [0] * num_opt_var


class mCHP:
    def __init__(self, num_opt_var):
        self.Out_mCHP = cp.Variable(num_opt_var)
        self.P_mCHP = cp.Variable(num_opt_var)
        self.Q_mCHP = cp.Variable(num_opt_var)
        self.C_mCHP = cp.Variable(num_opt_var)
        self.F_mCHP = cp.Variable(num_opt_var)


class CHP:
    def __init__(self, num_opt_var):
        self.C_CHP = cp.Variable(num_opt_var)
        self.F_CHP = cp.Variable(num_opt_var)
        self.P_CHP = cp.Variable(num_opt_var)
        self.Q_CHP = cp.Variable(num_opt_var)

        self.w11_CHP = cp.Variable(num_opt_var)
        self.w12_CHP = cp.Variable(num_opt_var)
        self.w13_CHP = cp.Variable(num_opt_var)
        self.w14_CHP = cp.Variable(num_opt_var)
        self.w21_CHP = cp.Variable(num_opt_var)
        self.w22_CHP = cp.Variable(num_opt_var)
        self.w23_CHP = cp.Variable(num_opt_var)
        self.w24_CHP = cp.Variable(num_opt_var)
        self.R_CHP = cp.Variable(1)
        self.D_CHP = cp.Variable(1)
        self.b_CHP = cp.Variable(num_opt_var, boolean=True)
        self.b1_CHP = cp.Variable(num_opt_var, boolean=True)
        self.b2_CHP = cp.Variable(num_opt_var, boolean=True)

        self.yon_CHP = cp.Variable(num_opt_var, boolean=True)
        self.zoff_CHP = cp.Variable(num_opt_var, boolean=True)
        self.ysum_CHP = cp.Variable(num_opt_var)
        self.zsum_CHP = cp.Variable(num_opt_var)


class GSHP:
    def __init__(self, num_opt_var):
        self.P_GSHP = cp.Variable(num_opt_var)
        self.Q_GSHP = cp.Variable(num_opt_var)


class GB:
    def __init__(self, num_opt_var):
        self.F_GB = cp.Variable(num_opt_var)
        self.C_GB = cp.Variable(num_opt_var)
        self.Q_GB = cp.Variable(num_opt_var)

        self.w0_GB = cp.Variable(num_opt_var)
        self.w1_GB = cp.Variable(num_opt_var)
        self.w2_GB = cp.Variable(num_opt_var)
        self.w3_GB = cp.Variable(num_opt_var)
        self.w4_GB = cp.Variable(num_opt_var)

        self.b_GB = cp.Variable(num_opt_var, boolean=True)
        self.b1_GB = cp.Variable(num_opt_var, boolean=True)
        self.b2_GB = cp.Variable(num_opt_var, boolean=True)
        self.b3_GB = cp.Variable(num_opt_var, boolean=True)
        self.b4_GB = cp.Variable(num_opt_var, boolean=True)


class Heat_Storage:
    def __init__(self, num_opt_var):
        self.Q_StorageCh = cp.Variable(num_opt_var)
        self.Q_StorageDc = cp.Variable(num_opt_var)
        self.Q_StorageTot = cp.Variable(num_opt_var)
        self.b_StorageCh = cp.Variable(num_opt_var, boolean=True)


class Elec_Storage:
    def __init__(self, num_opt_var):
        self.P_BatteryCh = cp.Variable(num_opt_var)
        self.P_BatteryDc = cp.Variable(num_opt_var)
        self.P_BatteryTot = cp.Variable(num_opt_var)
        self.b_BatteryCh = cp.Variable(num_opt_var, boolean=True)


class Elec_Grid:
    def __init__(self, num_opt_var):
        self.C_Grid = cp.Variable(num_opt_var)
        self.P_GridIn = cp.Variable(num_opt_var)
        self.P_GridOut = cp.Variable(num_opt_var)

        self.P_Slack = cp.Variable(num_opt_var)
        self.Q_Slack = cp.Variable(num_opt_var)


class trans_pw_ht:
    def __init__(self, num_opt_var, st, rt):
        self.P_strt = cp.Variable(num_opt_var)
        self.Q_strt = cp.Variable(num_opt_var)
        self.origin = st
        self.recpnt = rt


class trans_local:
    def __init__(self, num_opt_var):

        self.P_strt = cp.Variable(num_opt_var)
        self.Q_strt = cp.Variable(num_opt_var)
