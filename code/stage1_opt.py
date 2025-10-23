# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:04:36 2024

@author: SHI

stage1_ optimize sequence
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import get_variable_combination
import create_initial_position
import get_pot_cob
import stage2_opt
     
def optimize_sequence(L1,vehicles_info, phase_info, LastSignal, t0):
    for v in list(vehicles_info.keys()):
        if ( (vehicles_info[v]["i"]==1 and vehicles_info[v]["y"]>842 ) or
            (vehicles_info[v]["i"]==2 and vehicles_info[v]["x"]<0 ) or
            (vehicles_info[v]["i"]==3 and vehicles_info[v]["y"]<0 ) or
            (vehicles_info[v]["i"]==4 and vehicles_info[v]["x"]>842 )
            ): 
            vehicles_info.pop(v)
    #========= parameters
    ## gemetric parameters: ===
    L=200
    I = [1,2,3,4]
    K_swal = [1,2,3,4]
    
    ## vehicle parameters: ===
    V_max = len(vehicles_info)
    
    V_h_total = [v for v in list(vehicles_info.keys()) if vehicles_info[v]["type"]=="bus"]
    V_p_total = [v for v in list(vehicles_info.keys()) if vehicles_info[v]["type"]=="car"]
    V_total = V_h_total + V_p_total
      
    i_0 = {v: vehicles_info[v]["i"] for v in list(vehicles_info.keys())}
    k_0 = {v: 1 if vehicles_info[v]["lane"] == 4
           else 2 if vehicles_info[v]["lane"] == 5
           else 3 if vehicles_info[v]["lane"] == 6
           else 4 if vehicles_info[v]["lane"] == 7
           else 5 if vehicles_info[v]["lane"] == 1
           else 6 if vehicles_info[v]["lane"] == 2
           else 7
           for v in list(vehicles_info.keys())}
    x_0 = {v: (vehicles_info[v]["y"]-442)/2 if i_0[v] == 1 
           else L-vehicles_info[v]["x"]/2 if i_0[v] == 2 
           else L-vehicles_info[v]["y"]/2 if i_0[v] == 3
           else (vehicles_info[v]["x"]-442)/2
           for v in list(vehicles_info.keys())}
    
    
    for v in V_total:
        if x_0[v] > L1 and k_0[v] in K_swal:
            k_0[v] = 5 if k_0[v]==1 else 7 if k_0[v]==4 else 6
    
    free_flow_time = {v:vehicles_info[v]["free flow time"] for v in V_total}
    
    l_max = 11.11  # maximum distance per second
    #create_initial_position.plot_t0(i_0,k_0,x_0,V_h_total,V_p_total,L1,L)
    ## signal paramters: ===
    N = [1,2]   # the number of cycles
    T_max =N[-1]*120  # total time step  for signal
    ts = 0
    J = [1,2,3, 4,5,6, 7,8,9, 10,11,12]
    #Psi = J   # all phases
    Psi_e = [j for j in J if phase_info[j] == "ended"]   # ended phases
    Psi_a = [j for j in J if phase_info[j] == "active"]   # currently active phases
    Psi_i = [j for j in J if phase_info[j] == "inactive"]   # inactive phases
    
    # confliting phases
    Psi_ic_all = [(1,4),(1,5),(1,8),(1,9),(1,10),(1,11),
              (2,4),(2,5),(2,7),(2,10),(2,11),(2,12),
              (3,5),(3,7),
              (4,1),(4,2),(4,7),(4,8),(4,11),(4,12),
              (5,1),(5,2),(5,3),(5,7),(5,8),(5,10),
              (6,8),(6,10),
              
              (7,2),(7,3),(7,4),(7,5),(7,10),(7,11),
              (8,1),(8,4),(8,5),(8,6),(8,10),(8,11),
              (9,1),(9,11),
              (10,1),(10,2),(10,5),(10,6),(10,7),(10,8),
              (11,1),(11,2),(11,4),(11,7),(11,8),(11,9),
              (12,2),(12,4)]
    
    Psi_ic = [(1,4),(1,5),(1,8),(1,9),(1,10),(1,11),
              (2,4),(2,5),(2,7),(2,10),(2,11),(2,12),
              (3,5),(3,7),
              (4,7),(4,8),(4,11),(4,12),
              (5,7),(5,8),(5,10),
              (6,8),(6,10),
              
              (7,10),(7,11),
              (8,10),(8,11),
              (9,11)]
    
    G_min = 6
    phi = 4
    
    # lane and phase
    i_k_j = {}  
    i_k_j[(1,1)] = 1;i_k_j[(1,2)] = 2;i_k_j[(1,3)] = 2;i_k_j[(1,4)] = 3
    i_k_j[(2,1)] = 4;i_k_j[(2,2)] = 5;i_k_j[(2,3)] = 5;i_k_j[(2,4)] = 6
    i_k_j[(3,1)] = 7;i_k_j[(3,2)] = 8;i_k_j[(3,3)] = 8;i_k_j[(3,4)] = 9
    i_k_j[(4,1)] = 10;i_k_j[(4,2)] = 11;i_k_j[(4,3)] = 11;i_k_j[(4,4)] = 12
    
    #K_cwl = [1,2,3]       
    #d = 2

    ## auxiliary variable
    # vehicles of each arm
    V_i = {}
    for i in I:
        tmp = [v for v in list(i_0.keys()) if i_0[v] == i]
        V_i[i] = tmp
    
    # V_area initaol area 1: SWAL area 2:CWL
    V_area = {}
    V_area[1] = [v for v in V_total if x_0[v] <= L1]
    V_area[2] = [v for v in V_total if x_0[v] > L1] 

    # generating time
    t_g = {v: vehicles_info[v]['gen_time'] for v in V_total}
    #t_g = {v:0 for v in V_total}
        
    # the length of each vehicle
    l_v = {v: 5 for v in V_p_total}
    l_v.update({v: 12 for v in V_h_total})
    
    ##  potential combination_i
    # vehicles_segment in each lane
    Omega_ik = get_variable_combination.find_vehicles_in_each_lane(i_0,k_0,x_0,V_h_total)
    # vehicles in each area at time t0
    Omega_ia = get_variable_combination.find_vehicles_in_each_area(Omega_ik)  
    # potential combination
    fix_cob_i,pot_cob_i = get_pot_cob.find_potential_v_follow_SWAL(Omega_ik,V_h_total,V_p_total,x_0)
    pot_cob = [pot_cob_i[i][index] for i in I for index in range(len(pot_cob_i[i]))]
    fix_cob = [fix_cob_i[i][index] for i in I for index in range(len(fix_cob_i[i]))]
    pot_cob = list(set(pot_cob))
    fix_cob = list(set(fix_cob))
    
          
    ## vehicle lane  
    v_k_i = get_variable_combination.find_w_k_in_SWAL(Omega_ik,V_h_total,V_p_total)  
    v_k = []
    for i in I:
        v_k += v_k_i[i]
    # plot
    #create_initial_position.plot_t0(i_0,k_0,x_0,V_h_total,V_p_total,L1,L)
    
    
    # model ======================================================
    #  1. create model
    model = gp.Model('stage_sequence1')
    
    #  2. create variables
    # following variables
    alpha = model.addVars(pot_cob,vtype=GRB.BINARY, name="alpha_{w,w',k}")   # 0-1 variable 
    # leaving time
    leave = model.addVars(V_total,lb=t0/T_max,ub=1,vtype=GRB.CONTINUOUS, name="leave_{w}")  # ub=T_max
    
    ## signal constraints
    # cycle length
    C = model.addVars(N, lb=0, ub=1,vtype=GRB.CONTINUOUS ,name="C^n")  #ub=T_max
    # green start
    G = model.addVars(J,N,lb=0,ub=1,vtype=GRB.CONTINUOUS,name="G_{j,n}") #ub=T_max
    # green duration
    LAMBA = model.addVars(J,N,lb=G_min/T_max,ub=1,vtype=GRB.CONTINUOUS,name="LAMBDA_{j,n}")  # lb=G_min,ub=60
    # conflit phase
    OMEGA = model.addVars(Psi_ic_all,N,vtype=GRB.BINARY,name="omeage_{j,j'}^n")
    
    # vehicle choose cycle
    beta = model.addVars(V_total,N,vtype=GRB.BINARY, name="beta_{w,n}")
    
    # constraint1 car following relations =================
    # passenger car
        # each car can only follow one vehicle of all lanes
    model.addConstrs( sum(alpha[w,w_,k] for w_ in V_total+[0] for k in K_swal if (w,w_,k) in pot_cob) 
                      == 1 for w in V_p_total if w in V_area[2] )
    
        # each car can only be followed by one vehicles of all lanes
    model.addConstrs( sum(alpha[w_,w,k] for w_ in V_total+[1001,1002,1003,1004] for k in K_swal if (w_,w,k) in pot_cob) 
                      == 1 for w in V_p_total 
                      if (w in V_area[2]) or (w in [Omega_ik[i,k][-1] if len(Omega_ik[i,k]) > 0 else 0 for i in I for k in K_swal]) 
                      )
       
        # virtual vehicle 0 can only be followed by at most vehicles at each lane % <=1 means fixed_cob can follow 0
    model.addConstrs( sum(alpha[w_,0,k] for w_ in V_i[i]+[1000+i] if (w_,0,k) in pot_cob) 
                      <= 1 for i in I for k in K_swal)
    
        # virtual vehicle 100 can only follow one vehicle at each lane
    model.addConstrs( sum(alpha[1000+i,w,k] for w in V_i[i]+[0]  if (1000+i,w,k) in pot_cob) 
                      == 1 for i in I for k in K_swal)
    
    # heavy vehicle
        # can follow at most one vehicle at each permitted lane
    model.addConstrs( sum(alpha[w,w_,k] for w_ in V_i[i]+[0] if (w,w_,k) in pot_cob) 
                      <= 1 
                      for i in I for w in V_i[i] if (w in V_h_total) and (w in V_area[2])
                      for k in K_swal if (w,k) in v_k_i[i])
    
        # follow two vehicles at all permitted lanes
    model.addConstrs( sum(alpha[w,w_,k] for w_ in V_i[i]+[0] for k in K_swal if (w,w_,k) in pot_cob) 
                      == 2  
                      for i in I for w in V_i[i] if (w in V_h_total) and (w in V_area[2]) )    
    
        # can be followed by at most one vehicle at each permitted lane
    model.addConstrs( sum(alpha[w_,w,k] for w_ in V_i[i]+[1001,1002,1003,1004] if (w_,w,k) in pot_cob) 
                      <= 1 
                      for i in I for w in V_i[i] if (w in V_h_total) and (w in V_area[2])
                      for k in K_swal if (w,k) in v_k_i[i])
        # can be followed by two vehicles at all permitted lanes
    model.addConstrs( sum(alpha[w_,w,k] for w_ in V_i[i]+[1001,1002,1003,1004] for k in K_swal if (w_,w,k) in pot_cob) 
                      == 2 
                      for i in I for w in V_i[i] if (w in V_h_total) and (w in V_area[2]) ) 
    
        # the two lanes should be adjancant
    model.addConstrs( sum( alpha[w,w_,k_] for w_ in V_total+[0] for k_ in [k-1,k+1] if (w,w_,k_) in pot_cob)
                      >= sum( alpha[w,w_,k] for w_ in V_total+[0] if (w,w_,k) in pot_cob)
                      for (w,k) in v_k if (w in V_h_total) and (w in V_area[2]) and k in [2,3])
    
    model.addConstrs( sum( alpha[w,w_,2] for w_ in V_total+[0] if (w,w_,2) in pot_cob)
                      >= sum( alpha[w,w_,k] for w_ in V_total+[0] if (w,w_,k) in pot_cob)
                      for (w,k) in v_k if (w in V_h_total) and (w in V_area[2]) and k == 1)
    
    model.addConstrs( sum( alpha[w,w_,3] for w_ in V_total+[0] if (w,w_,3) in pot_cob)
                      >= sum( alpha[w,w_,k] for w_ in V_total+[0] if (w,w_,k) in pot_cob)
                      for (w,k) in v_k if (w in V_h_total) and (w in V_area[2]) and k == 4)
    
    # all vehicles
        # if the vehicle follow one vehicle in the k lane, it must be followed in the k lane
    model.addConstrs( sum(alpha[w2,w,k] for w2 in V_total+[1001,1002,1003,1004] if (w2,w,k) in pot_cob)
                     >= alpha[w,w1,k]
                     for (w,w1,k) in pot_cob if w1 !=0 and w not in [1001,1002,1003,1004])
    
        # the last vehicle in swal area should be followed
    model.addConstrs( sum(alpha[w_,Omega_ik[i,k][-1],k] for w_ in V_total+[1001,1002,1003,1004] 
                          if (w_,Omega_ik[i,k][-1],k) in pot_cob)
                      ==1 for i in I for k in K_swal if len(Omega_ik[i,k])>0)
    
    # signal constraints 
    # cycle length
    model.addConstr(C[1] >= (t0-ts)/T_max)  # (t0-ts)
    model.addConstr(sum(C[n] for n in N) <= 1)
    
    # green start
    model.addConstrs(ts/T_max+sum(C[n1] for n1 in N if n1 < n) <= G[j,n] for j in J for n in N)
    model.addConstrs(G[j,n] <= ts/T_max+sum(C[n1] for n1 in N if n1 <= n) for j in J for n in N)
        #- the green start is known if the phase was already started or ended
    model.addConstrs(G[j,1] == LastSignal[j][0]/T_max for j in Psi_e)
    model.addConstrs(G[j,1] == LastSignal[j][0]/T_max for j in Psi_a)
        #- the grren start for inactive phase should be larger than t0
    model.addConstrs(G[j,1] >= t0/T_max for j in Psi_i)
    
    # green duration
    model.addConstrs(LAMBA[j,n] <= C[n] for j in J for n in N)
        #- the green start is known if the phase was already ended
    model.addConstrs(LAMBA[j,1] == LastSignal[j][1]/T_max for j in Psi_e)
        #- the green duration that are green at time 𝑡0:
    model.addConstrs(LAMBA[j,1] >= t0/T_max - LastSignal[j][0]/T_max for j in Psi_a)
    
    
    # green end
    model.addConstrs(ts/T_max + sum(C[n1] for n1 in N if n1 < n) <= G[j,n] + LAMBA[j,n] for j in J for n in N)
    model.addConstrs(G[j,n] + LAMBA[j,n] <= ts/T_max +sum(C[n1] for n1 in N if n1 <= n) for j in J for n in N)
    
    # conflict movements
    model.addConstrs(OMEGA[j,j1,n] + OMEGA[j1,j,n] == 1 for (j,j1) in Psi_ic for n in N)
    
    model.addConstrs(G[j,n] + LAMBA[j,n] + phi/T_max <= G[j1,n] + 2*(1-OMEGA[j,j1,n]) 
                     for (j,j1) in Psi_ic for n in N)
    model.addConstrs(G[j1,n] + LAMBA[j1,n] + phi/T_max <= G[j,n] + 2*(1-OMEGA[j1,j,n]) 
                     for (j,j1) in Psi_ic for n in N)
    
    model.addConstrs(G[j1,n] + LAMBA[j1,n] + phi/T_max <= G[j1,n+1]
                     for (j,j1) in Psi_ic_all for n in N[0:-1])
    
    # cycle selection
    model.addConstrs(sum(beta[w,n] for n in N) == 1 for w in V_total)
    
    ## leave time
    # leave time related to sequence
    model.addConstrs( leave[w] - leave[w_] - 1.5/T_max >= -2*(1-alpha[w,w_,k]) 
                     for (w,w_,k) in pot_cob if w_!=0 and w not in [1001,1002,1003,1004] and w_ in V_p_total)
    
    model.addConstrs( leave[w] - leave[w_] - 3/T_max >= -2*(1-alpha[w,w_,k]) 
                     for (w,w_,k) in pot_cob if w not in [1001,1002,1003,1004] and w_ in V_h_total) 
    
    model.addConstrs( leave[w] - leave[w_] >= 1.5/T_max
                     for (w,w_,k) in fix_cob if w_!=0 and w not in [1001,1002,1003,1004] and w_ in V_p_total)
    
    model.addConstrs( leave[w] - leave[w_] >= 3/T_max
                     for (w,w_,k) in fix_cob if w_!=0 and w not in [1001,1002,1003,1004] and w_ in V_h_total)
    
    # leave time related to free flow and lane changing
    model.addConstrs( leave[w] >= x_0[w]/l_max/T_max + t0/T_max for w in V_total)    
    V_can_change =  [v for i in I for v in Omega_ik[i,6] if v in V_h_total and x_0.get(v, float('inf')) >= L1+30]

    model.addConstrs( leave[w]  >= G[i_k_j[i,k],n] 
                                   -2 *(1- sum(alpha[w,w_,k] for w_ in V_i[i]+[0] if (w,w_,k) in pot_cob) )
                                   -2 *(1-beta[w,n])
                      for i in I for w in V_i[i] if w in V_area[2] 
                      for k in K_swal if (w,k) in v_k_i[i] for n in N)
    
    model.addConstrs( leave[w] <= G[i_k_j[i,k],n] + LAMBA[i_k_j[i,k],n] 
                                  + 2*(1- sum(alpha[w,w_,k] for w_ in V_i[i]+[0] if (w,w_,k) in pot_cob) )
                                  + 2*(1-beta[w,n])
                      for i in I for w in V_i[i] if w in V_area[2] 
                      for k in K_swal if (w,k) in v_k_i[i] for n in N)
    
        # -fixed cob
    model.addConstrs( leave[w]  >= G[i_k_j[i,k],n] -2 *(1-beta[w,n]) 
                      for i in I for k in K_swal for w in Omega_ik[i,k] for n in N)
    
    model.addConstrs( leave[w] <= G[i_k_j[i,k],n] + LAMBA[i_k_j[i,k],n] + 2*(1-beta[w,n])
                      for i in I for k in K_swal for w in Omega_ik[i,k] for n in N)

    # 5.objective
    obj = sum(leave[w]-t_g[w]/T_max-free_flow_time[w]/T_max for w in V_total)     #不是x0 是更远的地方          
    model.setObjective(obj,GRB.MINIMIZE)
    # optimize
    model.setParam('Presolve', 2)      # 开启高级预处理
    model.setParam('Heuristics', 0.5)  # 增加启发式求解的权重
    model.setParam('MIPFocus', 1)      # 优先寻找可行解
    model.Params.TimeLimit = 1.5

    model.optimize()
    
    #print('status',GRB.Status.OPTIMAL)
    #model.printQuality()    
    if model.status == gp.GRB.OPTIMAL:
        print("find optimal result！")
        print("optimal result:" ,model.objVal/V_max*T_max)
        #
        result = pd.DataFrame()
        var_name = []
        var_value = []
        for v in model.getVars():
            #print('%s %g' % (v.varName, v.x))
            var_name.append(v.varName)
            var_value.append(v.x)
        result['var'] = var_name
        result['value'] = var_value    

        data = result
        # signal
        signal = data[data['var'].str.contains('G_{')] 
        signal.loc[:, 'j'] = signal['var'].apply(lambda r:int(r.split("[")[-1].split(",")[0]))
        signal.loc[:, 'n'] = signal['var'].apply(lambda r:int(r.split(",")[-1].split("]")[0]))

        lamba = data[data['var'].str.contains('LAMBDA_{')] 
        lamba.loc[:, 'j'] = lamba['var'].apply(lambda r:int(r.split("[")[-1].split(",")[0]))
        lamba.loc[:, 'n'] = lamba['var'].apply(lambda r:int(r.split(",")[-1].split("]")[0]))

        signal = pd.merge(signal,lamba,on=['j','n'],how='left')

        signal = signal[['j','n','value_y','value_x']]
        signal = signal.rename(columns={"value_y":"lamba","value_x":'start'})
        signal.loc[:, 'end'] = signal.apply(lambda r: r['start']+r['lamba'],axis=1)

        signal.loc[:, 'start'] = signal['start'].apply(lambda r:round(r*T_max,2))
        signal.loc[:, 'lamba'] = signal['lamba'].apply(lambda r:round(r*T_max,2))
        signal.loc[:, 'end'] = signal['end'].apply(lambda r:round(r*T_max,2))
        
        signal_tmp = {j:(signal[ (signal["j"]==j) & (signal["n"]==1) ] ["start"].iloc[0],
                         signal[ (signal["j"]==j) & (signal["n"]==1) ] ["lamba"].iloc[0],
                         phi) 
                      for j in range(1,9+4)}
        signal = signal_tmp
        
        # alpha
        alpha = data[data['var'].str.contains('alpha_{')] 
        alpha['value'] = alpha['value'].apply(lambda r:round(r))
        alpha = alpha[alpha['value']==1]

        alpha.loc[:, 'w'] = alpha['var'].apply(lambda r: int(r.split("[")[-1].split(",")[0]))
        alpha.loc[:, 'w_'] = alpha['var'].apply(lambda r: int(r.split("[")[-1].split(",")[1]))
        alpha.loc[:, 'k'] = alpha['var'].apply(lambda r:int(r.split("[")[-1].split(",")[-1][0:-1]))

        alpha = alpha[["w","w_","k"]]

        for cob in fix_cob:
            new_rows = pd.DataFrame(fix_cob, columns=['w', 'w_', 'k'])
            alpha = pd.concat([alpha, new_rows], ignore_index=True)
            
        alpha = alpha[alpha["w"].isin(V_area[2]+[1001,1002,1003,1004])]
        alpha.loc[:, 'x'] = alpha['w'].apply(lambda r: x_0[r] if r not in [1001,1002,1003,1004] else 200)
        alpha.loc[:, 'i'] = alpha['w'].apply(lambda r: int(i_0[r]) if (r!= 0 and r not in [1001,1002,1003,1004]) else 5)
        alpha = alpha.sort_values(by=['i','k','w','x'],ascending=True)
        
        # 2. find lane changed vehicles
        #V_can_change =  [v for i in I for v in Omega_ik[i,6] if v in V_h_total]

        V_change = []
        Target_lane = {}
        Forward_v = {}
        Back_v = {}
        x0 = {}
        lv = {}
        
        # 只保留距离 >= 110 的
        V_can_change = [v for v in V_can_change if x_0.get(v, float('inf')) >= L1+30]

                

        for v in V_can_change:
            x0[v] = x_0[v]
            lv[v] = l_v[v]
            
            tmp = alpha[alpha['w'] == v]
            tmp1 = alpha[alpha['w_'] == v]
            
            if 1 in list(tmp['k']):
                target_lane = 5
                forward = tmp[tmp['k']==1]['w_'].iloc[0]
                back = tmp1[tmp1['k']==1]['w'].iloc[0]
            elif 4 in list(tmp['k']):
                target_lane = 7
                forward = tmp[tmp['k']==4]['w_'].iloc[0]
                back = tmp1[tmp1['k']==4]['w'].iloc[0]
            else:
                target_lane = 6
                forward = None
                back = None
            
            if target_lane != 6:
                V_change.append(v)
                Target_lane[v] = target_lane
                
                if forward in [v for i in I for v in Omega_ia[i,2]]:
                    Forward_v[v] = forward
                    x0[forward] = x_0[forward]
                    lv[forward] = l_v[forward]
                else:
                    Forward_v[v] = None
                
                if back in  [v for i in I for v in Omega_ia[i,2]]:
                    Back_v[v] = back
                    x0[back] = x_0[back]
                    lv[back] = l_v[back]
                else:
                    Back_v[v] = None

        Lane_Change_Time,Speed = stage2_opt.Optimize_Lane_Change(L1,V_change,Target_lane, Forward_v, Back_v, x_0, k_0, i_0, V_h_total)
        
        return model.objVal/V_max*T_max, signal, Target_lane, Lane_Change_Time,Speed
        
    else:
        print("can not find optimal result within limited time")
        return None, None, {}, {},{}
    
